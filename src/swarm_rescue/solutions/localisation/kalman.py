"""
kalman.py : filtre de Kalman "fusion" pour l'estimation de position du drone.

Ce module est le filtre qui a gagné l'expérience « lignes droites » du
notebook Test_Kalman_lignes_droites.ipynb (MSE en mouvement ~38 px² contre
~53 px² pour le GPS brut, seed 42, gain ~28 %). Il n'a aucune dépendance
au simulateur : seulement numpy. On peut donc l'appeler depuis n'importe
quelle classe de drone du dossier solutions/, ou le tester seul.

Principe (filtre à état augmenté, méthode Bryson-Henrikson) :

    état    X = [x, y, vx, vy, bx, by]
    position et vitesse à modèle constant (Px = Px + Vx*dt)
    biais GPS (bx, by) modélisé en auto-régressif d'ordre 1 :
        b_t = PHI * b_{t-1} + w_t

    Le bruit du GPS du simulateur est auto-régressif d'ordre 1 (dérive,
    correlation temporelle), PAS blanc : c'est modélisé par les états de
    biais. La mesure GPS observe alors position + biais.

    La vitesse mesurée par l'odomètre + le compas (drone.measured_velocity(),
    en px/pas) observe directement [vx, vy], indépendamment du biais GPS :
    elle rend la vitesse observable et le filtre identifiable.

    Bruit de la vitesse mesurée (point clé, cf. l'expérience) : la distance
    odométrique est excellente (bruit blanc ~0.2 px/pas), mais la direction
    cumule l'angle odomètre (8°, blanc) et le compas (4°, AR(1)), soit
    ~0.156 rad RMS : l'erreur latérale vaut ~0.156 * |v| px/pas, donc
    proportionnelle à la vitesse. Un sigma fixe (0.4 px/pas calibré sur un
    drone immobile) rend le filtre pire que le GPS brut dès 30 px/pas.
    D'où la variance adaptative recalculée à chaque pas :

        sigma_v² = SIGMA_VEL_MIN² + (KANG * |v_mes|)²

Interface principale :

    kf = KalmanFusion()
    pos = kf.step(gps, v_meas)   # -> np.array([x, y]) ou None

    gps    : ping GPS bruité, array-like [x, y] en px, ou None.
    v_meas : vitesse mesurée, array-like [vx, vy] en px/pas, ou None.

    - GPS seul dispo  : correction GPS uniquement (mode dégradé) ;
    - odomètre seul   : prédiction + correction vitesse uniquement
                        (odométrie : utile dans une zone sans GPS, la
                        covariance de position croît à chaque pas) ;
    - rien du tout    : prédiction seule ;
    - jamais initialisé et pas de GPS : renvoie None.

Utilisation typique dans une classe de drone (cf. control() ci-dessous) :

    from swarm_rescue.solutions.localisation.kalman import KalmanFusion

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.kalman = KalmanFusion()

    def control(self):
        self.position_estimate_ = self.kalman.step(
            self.measured_gps_position(),   # ping GPS bruité, px
            self.measured_velocity(),       # px/pas, odomètre + compas
        )
        # ... puis commander le drone sur self.position_estimate_
        # au lieu du GPS brut.

On peut aussi utiliser kf.step_sensors(drone), qui lit seul les deux
capteurs du drone passé en argument.

Ne jamais utiliser true_position() dans control() : interdit par le garde
du simulateur (TrueStateAccessError). Ce filtre n'utilise QUE les
capteurs bruités, il respecte donc la règle.

Chiffres de référence (expérience lignes droites, seed 42, MSE position
en mouvement, seuil 0.5 px/pas) :
    GPS brut        53.0 px²
    KF basique      56.6 px²
    KF augmenté     53.3 px²
    KF fusion       38.3 px²   <- ce filtre
"""

import numpy as np

__all__ = ["KalmanFusion"]


class KalmanFusion:
    """
    Filtre de Kalman à état augmenté + fusion odomètre (GPS AR(1)).

    État : [x, y, vx, vy, bx, by] en px et px/pas (dt = 1 pas de simulateur).
    La position estimée est la partie [x, y] : le filtre retranche le biais
    GPS tout seul.

    Paramètres des capteurs du simulateur (cf. drone_sensors.py) :
        PHI = 0.98, std stationnaire GPS 5 px/axe ;
        odomètre : distance blanche 0.2 px/pas, angle blanc 8° ;
        compas : AR(1), std 4°.
    """

    # --- paramètres du filtre (valeurs validées sur 3 seeds, nov. 2026) ---
    PHI = 0.98  # paramètre AR(1) du bruit GPS du simulateur
    SIGMA_GPS = 5.0  # std stationnaire du bruit GPS (px)
    SIGMA_ACCEL = 4.0  # bruit d'accélération du modèle (px/pas²) :
    # accélérations réelles jusqu'à ~6 px/pas² (F=300, m=50)
    SIGMA_VEL_MIN = 0.2  # bruit blanc de la distance odomètre (px/pas)
    KANG = 0.156  # erreur de direction RMS de la vitesse mesurée (rad)
    # = sqrt((8°)² + (4°)²) ≈ 0.156

    _DIM = 6

    def __init__(self):
        dt = 1.0
        # std du bruit blanc qui fait vivre le biais AR(1) (~1 px)
        self.sigma_b = self.SIGMA_GPS * np.sqrt(1.0 - self.PHI**2)
        # matrice de transition : position += vitesse*dt, biais en AR(1)
        self.F = np.eye(self._DIM)
        self.F[0, 2] = dt
        self.F[1, 3] = dt
        self.F[4, 4] = self.PHI
        self.F[5, 5] = self.PHI
        # bruit de modèle : accélération sur (x, y) + bruit blanc du biais
        G = np.zeros((self._DIM, 4))
        G[0, 0] = 0.5 * dt * dt
        G[2, 0] = dt
        G[1, 1] = 0.5 * dt * dt
        G[3, 1] = dt
        G[4, 2] = 1.0
        G[5, 3] = 1.0
        gains = np.array(
            [self.SIGMA_ACCEL, self.SIGMA_ACCEL, self.sigma_b, self.sigma_b]
        )
        self.Q = G @ np.diag(gains**2) @ G.T
        # observation GPS : le ping observe position + biais
        self.H_gps = np.array(
            [[1.0, 0.0, 0.0, 0.0, 1.0, 0.0], [0.0, 1.0, 0.0, 0.0, 0.0, 1.0]]
        )
        self.R_gps = (0.2**2) * np.eye(2)  # mesure quasi sans bruit blanc
        # observation vitesse (odomètre + compas) : observe [vx, vy]
        self.H_vel = np.array(
            [[0.0, 0.0, 1.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 1.0, 0.0, 0.0]]
        )
        self.I6 = np.eye(self._DIM)

        # état et covariance : initialisés au premier ping GPS
        self.x = None
        self.P = np.diag(
            [
                self.SIGMA_GPS**2,
                self.SIGMA_GPS**2,
                4.0,
                4.0,
                self.SIGMA_GPS**2,
                self.SIGMA_GPS**2,
            ]
        )
        self.pos = None  # dernière estimation [x, y]

    # ------------------------------------------------------------------
    # utilitaires
    # ------------------------------------------------------------------
    @staticmethod
    def _sanitize(v):
        """Array numpy [2] si v est fini, sinon None."""
        if v is None:
            return None
        v = np.asarray(v, dtype=float).reshape(-1)
        if v.size != 2 or not np.all(np.isfinite(v)):
            return None
        return v

    def reset(self):
        """Remet le filtre à zéro (prochain ping = réinitialisation)."""
        self.__init__()

    # ------------------------------------------------------------------
    # étapes du filtre
    # ------------------------------------------------------------------
    def _predict(self):
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q

    def _correct(self, z, H, R):
        """Correction de Kalman : gain, mise à jour état et covariance."""
        innovation = z - H @ self.x
        S = H @ self.P @ H.T + R
        K = self.P @ H.T @ np.linalg.inv(S)
        self.x = self.x + K @ innovation
        self.P = (self.I6 - K @ H) @ self.P

    def _correct_velocity(self, v_meas):
        """Correction par la seule vitesse odométrique (zone sans GPS).

        sigma_v adaptatif : l'erreur de direction de la vitesse mesurée
        vaut ~KANG*|v| px/pas (angle odomètre 8° + compas 4°), on ne peut
        pas la supposer constante.
        """
        vmag = float(np.hypot(v_meas[0], v_meas[1]))
        sv2 = self.SIGMA_VEL_MIN**2 + (self.KANG * vmag) ** 2
        self._correct(v_meas, self.H_vel, sv2 * np.eye(2))

    def _correct_gps_velocity(self, gps, v_meas):
        """Correction jointe GPS + vitesse, mesures empilées (mode normal)."""
        vmag = float(np.hypot(v_meas[0], v_meas[1]))
        sv2 = self.SIGMA_VEL_MIN**2 + (self.KANG * vmag) ** 2
        Hf = np.vstack([self.H_gps, self.H_vel])
        Rf = np.diag([self.R_gps[0, 0], self.R_gps[1, 1], sv2, sv2])
        self._correct(np.concatenate([gps, v_meas]), Hf, Rf)

    # ------------------------------------------------------------------
    # interface principale
    # ------------------------------------------------------------------
    def step(self, gps=None, v_meas=None):
        """
        Un pas de filtrage : à appeler une fois par pas de simulateur,
        au début de control(), avant la commande.

        Args:
            gps: ping GPS bruité [x, y] en px (measured_gps_position()),
                 ou None si le GPS est indisponible.
            v_meas: vitesse mesurée [vx, vy] en px/pas
                    (measured_velocity(), odomètre + compas), ou None.

        Returns:
            np.array [x, y]: estimation de la position réelle (px),
            ou None si le filtre n'a encore jamais vu de GPS.
        """
        gps = self._sanitize(gps)
        v_meas = self._sanitize(v_meas)

        # -- initialisation : il faut un premier GPS pour partir ----------
        if self.x is None:
            if gps is None:
                return None
            self.x = np.array([gps[0], gps[1], 0.0, 0.0, 0.0, 0.0])
            if v_meas is not None:
                self.x[2] = v_meas[0]
                self.x[3] = v_meas[1]
            self.pos = np.array([self.x[0], self.x[1]])
            return self.pos.copy()

        # -- prédiction ----------------------------------------------------
        self._predict()

        # -- correction selon les capteurs disponibles ---------------------
        if gps is not None and v_meas is not None:
            self._correct_gps_velocity(gps, v_meas)
        elif gps is not None:
            self._correct(gps, self.H_gps, self.R_gps)
        elif v_meas is not None:
            # pas de GPS : odométrie seule (le biais GPS décroît vers 0,
            # la covariance de position croît à chaque pas)
            self._correct_velocity(v_meas)
        # sinon : prédiction seule

        self.pos = np.array([self.x[0], self.x[1]])
        return self.pos.copy()

    def step_sensors(self, drone):
        """
        Variante pratique : lit les capteurs du drone passé en argument.

        pos = kf.step_sensors(drone)

        Utilise drone.measured_gps_position() et drone.measured_velocity();
        aucune import du simulateur ici : le module reste utilisable seul.
        """
        gps = None
        v = None
        if hasattr(drone, "measured_gps_position"):
            gps = drone.measured_gps_position()
        if hasattr(drone, "measured_velocity"):
            v = drone.measured_velocity()
        return self.step(gps, v)

    # ------------------------------------------------------------------
    # lectures de l'état
    # ------------------------------------------------------------------
    def position(self):
        """Estimation courante [x, y] en px, ou None."""
        return None if self.pos is None else self.pos.copy()

    def velocity(self):
        """Estimation courante [vx, vy] en px/pas, ou None."""
        return None if self.x is None else np.array([self.x[2], self.x[3]])

    def gps_bias(self):
        """Estimation du biais GPS [bx, by] en px, ou None."""
        return None if self.x is None else np.array([self.x[4], self.x[5]])

    def covariance(self):
        """Matrice de covariance P (6x6) de l'état courant."""
        return None if self.x is None else self.P.copy()


# ----------------------------------------------------------------------
# petit auto-test : python3 kalman.py
# 1) vérifie le câblage du filtre sur des mesures synthétiques (bruits
#    calqués sur drone_sensors.py) : pas de crash, capteurs manquants
#    bien gérés, jamais de divergence catastrophique ;
# 2) affiche MSE GPS brut vs MSE filtrée à titre indicatif : le gain
#    réel dépend des capteurs exacts du simulateur, il est validé sur
#    le run du notebook (38.3 vs 53.0 px², cf. en-tête du module).
# ----------------------------------------------------------------------
if __name__ == "__main__":
    rng = np.random.default_rng(42)

    # bruits capteurs (cf. drone_sensors.py)
    SIGMA_GPS, PHI_G = 5.0, 0.98  # GPS AR(1), std stationnaire 5 px
    SIGMA_DIST = 0.2  # odomètre : distance blanche, px/pas
    SIGMA_ALPHA = np.deg2rad(8.0)  # odomètre : angle du déplacement
    SIGMA_COMPASS, PHI_C = np.deg2rad(4.0), 0.98  # compas AR(1), 4°
    W_GPS = SIGMA_GPS * np.sqrt(1.0 - PHI_G**2)  # bruit blanc du biais AR(1)
    W_COMPASS = SIGMA_COMPASS * np.sqrt(1.0 - PHI_C**2)

    gps_bias = np.zeros(2)  # biais AR(1) du GPS
    compass_bias = 0.0  # biais AR(1) du compas

    kf = KalmanFusion()
    pos = np.zeros(2)  # position vraie (px)
    vel = np.zeros(2)  # vitesse vraie (px/pas)
    heading = 0.0
    err_raw, err_kf = [], []

    for t in range(5000):
        # trajectoire en mouvement : virages brefs + croisière (~25 px/pas)
        if t % 400 < 50:
            heading += 0.03
        vel = (vel + 1.0 * np.array([np.cos(heading), np.sin(heading)])) * 0.96
        pos = pos + vel

        # --- capteurs bruités (un pas = un pas de simulateur) ---
        gps_bias = PHI_G * gps_bias + rng.normal(0.0, W_GPS, 2)
        gps_fake = pos + gps_bias
        compass_bias = PHI_C * compass_bias + rng.normal(0.0, W_COMPASS)
        theta = np.arctan2(vel[1], vel[0]) + compass_bias + rng.normal(0.0, SIGMA_ALPHA)
        speed_meas = np.hypot(vel[0], vel[1]) + rng.normal(0.0, SIGMA_DIST)
        v_meas = abs(speed_meas) * np.array([np.cos(theta), np.sin(theta)])

        # --- filtre ---
        est = kf.step(gps_fake, v_meas)
        if est is not None and np.hypot(vel[0], vel[1]) > 0.5:
            err_raw.append(np.sum((gps_fake - pos) ** 2))
            err_kf.append(np.sum((est - pos) ** 2))

    mse_raw = np.mean(err_raw)
    mse_kf = np.mean(err_kf)
    print(f"MSE GPS brut (synthétique) : {mse_raw:8.1f} px2")
    print(f"MSE Kalman fusion          : {mse_kf:8.1f} px2")
    print(f"gain                       : {100 * (1 - mse_kf / mse_raw):5.1f} %")
    # câblage correct : jamais de divergence même sur mesures imparfaites
    assert (
        np.isfinite(mse_kf) and mse_kf < 4 * mse_raw + 100
    ), "le filtre diverge (bug de câblage)"

    # --- cas limites ---
    kf2 = KalmanFusion()
    assert (
        kf2.step(None, np.array([1.0, 0.0])) is None
    ), "sans premier GPS, le filtre ne peut pas s'initialiser"
    assert (
        kf2.step(np.array([3.0, 4.0]), None) is not None
    ), "GPS seul doit fonctionner (mode dégradé)"
    est_nan = kf2.step(np.array([np.nan, 0.0]), np.array([0.0, 0.0]))
    assert est_nan is not None, "un NaN doit être ignoré sans crash"
    for _ in range(50):  # phase sans GPS : odométrie seule
        est = kf2.step(None, np.array([2.0, 0.0]))
        assert est is not None, "odométrie seule doit continuer"
    assert np.all(np.isfinite(kf2.position()))
    assert kf2.velocity() is not None and kf2.gps_bias() is not None
    kf2.reset()
    assert kf2.position() is None
    print("câblage : OK (init, GPS seul, NaN, sans GPS, reset)")
