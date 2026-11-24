"""
Filtre de Kalman simple pour Swarm Rescue, adapté du TP3
(Filtrage de Kalman et suivi de trajectoires, Telecom SudParis).

Modele identique au TP : etat x = [px, vx, py, vy], dynamique a vitesse
constante, capteur qui mesure la position. Le cas NaN du TP (question 12)
devient None : measured_gps_position() renvoie None en zone sans GPS.

A la fin du launcher, une figure style TP s'ouvre pour chaque drone
trace : vraie trajectoire, GPS observe, estimation Kalman, et le MSE
total dans le titre du troisieme panneau.

La vraie trajectoire n'est accessible qu'en mode debug, avec la variable
d'environnement SWARM_RESCUE_ALLOW_TRUE_STATE=1. Sans elle, la figure
montre GPS et estimation, sans MSE. Supprimer ce bloc debug avant la
soumission, conformement aux submission_guidelines.md.
"""

import atexit

import matplotlib.pyplot as plt
import numpy as np

from swarm_rescue.simulation.drone.drone_abstract import DroneAbstract
from swarm_rescue.simulation.utils.true_state_guard import true_state_is_allowed


class KalmanFilter:
    """Filtre de Kalman a vitesse constante, 4 etats : [px, vx, py, vy].

    kallman_iter(y) fait une prediction systematique, puis une mise a
    jour si y est une mesure GPS valide. y None (zone sans GPS) donne
    une prediction seule, comme le cas NaN du TP.
    """

    def __init__(
        self, Te=1.0, sigma_Q=1.0, sigma_px=5.0, sigma_py=5.0, x_init=None, P_init=None
    ):
        self.F = np.array(
            [
                [1.0, Te, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, Te],
                [0.0, 0.0, 0.0, 1.0],
            ]
        )
        self.Q = sigma_Q**2 * np.array(
            [
                [Te**3 / 3, Te**2 / 2, 0.0, 0.0],
                [Te**2 / 2, Te, 0.0, 0.0],
                [0.0, 0.0, Te**3 / 3, Te**2 / 2],
                [0.0, 0.0, Te**2 / 2, Te],
            ]
        )
        self.H = np.array([[1.0, 0.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0]])
        self.R = np.array([[sigma_px**2, 0.0], [0.0, sigma_py**2]])
        self.x = np.zeros(4) if x_init is None else np.asarray(x_init, dtype=float)
        self.P = np.eye(4) if P_init is None else np.asarray(P_init, dtype=float)

    def kallman_iter(self, y):
        """Prediction, puis mise a jour si la mesure GPS est disponible.

        Args:
            y: vecteur [x, y] mesure par le GPS, ou None en zone sans GPS.

        Returns:
            l'etat estime [px, vx, py, vy] apres cette etape.
        """
        # Prediction
        x_pred = self.F @ self.x
        P_pred = self.Q + self.F @ self.P @ self.F.T

        if y is None:  # zone sans GPS : prediction seule
            self.x, self.P = x_pred, P_pred
        else:
            # Mise a jour
            y_hat = y - self.H @ x_pred
            S = self.H @ P_pred @ self.H.T + self.R
            K = P_pred @ self.H.T @ np.linalg.inv(S)
            self.x = x_pred + K @ y_hat
            self.P = P_pred - K @ self.H @ P_pred

        return self.x


def calc_mse(estimated, true):
    """Erreur quadratique moyenne entre deux listes de positions.

    Meme esprit que calc_MSE du TP, applique aux positions seulement :
    moyenne des carres des ecarts, composante par composante.
    """
    est = np.asarray(estimated, dtype=float)
    tru = np.asarray(true, dtype=float)
    n = min(len(est), len(tru))
    return float(np.mean((est[:n] - tru[:n]) ** 2))


class MyDroneKalman(DroneAbstract):
    """Drone qui estime sa position par filtre de Kalman sur le GPS.

    Trace en memoire la trajectoire estimee, le GPS brut, et la vraie
    position en mode debug. A la fin du launcher, une figure style TP
    s'ouvre pour chaque drone trace.
    """

    # Nombre de figures affichees a la fin : les N derniers drones traces.
    # En essaim de 10 drones, limiter evite 10 fenetres par round.
    PLOT_MAX_DRONES = 3

    _instances = []  # tous les drones crees, pour le trace final

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._filter = None
        self.estimated_position = None
        self.log_estimate = []  # trajectoire estimee, un point par step
        self.log_gps = []  # GPS brut, un point par step
        self.log_true = []  # vraie position, mode debug uniquement
        MyDroneKalman._instances.append(self)

    def define_message_for_all(self) -> None:
        """Methode abstraite obligatoire : le launcher l'exige."""
        pass

    def control(self):
        gps = self.measured_gps_position()

        # Initialisation du filtre au premier point GPS valide
        if self._filter is None and gps is not None:
            self._filter = KalmanFilter(
                x_init=[gps[0], 0.0, gps[1], 0.0],
                # incertitude initiale : variance du GPS (25) sur la
                # position, vitesse inconnue (100)
                P_init=np.diag([25.0, 100.0, 25.0, 100.0]),
            )

        if self._filter is not None:
            y = None if gps is None else np.array([gps[0], gps[1]])
            x_est = self._filter.kallman_iter(y)
            self.estimated_position = (x_est[0], x_est[1])
            self.log_estimate.append([x_est[0], x_est[1]])
        else:
            # pas encore de point GPS : on garde les logs alignes
            self.log_estimate.append([np.nan, np.nan])

        # GPS brut : toujours legal dans control(), trace systematique
        if gps is not None:
            self.log_gps.append([gps[0], gps[1]])
        else:
            self.log_gps.append([np.nan, np.nan])

        # Vraie position : uniquement en mode debug, quand le garde-fou
        # sur l'etat vrai est volontairement desactive.
        if true_state_is_allowed():
            true_pos = self.true_position()
            self.log_true.append([true_pos[0], true_pos[1]])

        # Commande provisoire : avancer doucement pour exercer le filtre
        return {"forward": 0.5, "lateral": 0.0, "rotation": 0.0, "grasper": 0}


def _plot_drone(drone):
    """Figure style TP : 3 panneaux, vraie position, GPS, Kalman, MSE."""
    est = np.asarray(drone.log_estimate, dtype=float)
    gps = np.asarray(drone.log_gps, dtype=float)
    has_true = len(drone.log_true) == len(est) and len(est) > 0
    tru = np.asarray(drone.log_true, dtype=float) if has_true else None

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle(f"Drone {getattr(drone, 'identifier', '?')}")

    # Panneau 1 : vraie trajectoire et GPS observe
    axes[0].scatter(gps[:, 0], gps[:, 1], c="r", s=8, alpha=0.66, label="GPS observe")
    if tru is not None:
        axes[0].scatter(
            tru[:, 0], tru[:, 1], c="b", s=8, alpha=0.66, label="vraie position"
        )
    axes[0].set_title("Vraie et observee")
    axes[0].legend()

    # Panneau 2 : estimation du filtre de Kalman
    axes[1].scatter(
        est[:, 0], est[:, 1], c="g", s=8, alpha=0.66, label="filtre de Kalman"
    )
    axes[1].set_title("Estimee (Kalman)")
    axes[1].legend()

    # Panneau 3 : les trois traces et le MSE total
    axes[2].scatter(gps[:, 0], gps[:, 1], c="r", s=8, alpha=0.66, label="GPS observe")
    axes[2].scatter(
        est[:, 0], est[:, 1], c="g", s=8, alpha=0.66, label="filtre de Kalman"
    )
    if tru is not None:
        axes[2].scatter(
            tru[:, 0], tru[:, 1], c="b", s=8, alpha=0.66, label="vraie position"
        )
        mse = calc_mse(est, tru)
        axes[2].set_title(f"MSE total : {mse:.2e}")
    else:
        axes[2].set_title(
            "MSE indisponible : lancer avec SWARM_RESCUE_ALLOW_TRUE_STATE=1"
        )
    axes[2].legend()
    fig.tight_layout()


@atexit.register
def _plot_all_at_exit():
    """A la fin du launcher, trace les derniers drones en memoire.

    Les figures s'ouvrent quand la simulation se termine : fermez-les
    pour laisser le processus se terminer.
    """
    drones = [d for d in MyDroneKalman._instances if len(d.log_estimate) > 1]
    for drone in drones[-MyDroneKalman.PLOT_MAX_DRONES :]:
        try:
            _plot_drone(drone)
        except Exception as err:  # le trace ne doit jamais casser le run
            print(f"[kalman] trace impossible : {err}")
    if drones:
        plt.show()


def offline_test(sigma_Q=1.0, sigma_gps=5.0, N=200, blackout=(60, 90), seed=0):
    """Validation hors simulateur, dans l'esprit du TP.

    Genere une trajectoire vraie avec le modele exact du filtre, des
    observations GPS bruitees, une fenetre sans GPS, puis compare le MSE
    du filtre au MSE du GPS brut :

        uv run python src/swarm_rescue/solutions/my_drone_kalman.py
    """
    rng = np.random.default_rng(seed)

    template = KalmanFilter(sigma_Q=sigma_Q, sigma_px=sigma_gps, sigma_py=sigma_gps)
    F, Q, H, R = template.F, template.Q, template.H, template.R

    # Trajectoire vraie, equivalent de creer_trajectoire du TP
    x_true = np.zeros((N, 4))
    x_true[0] = [0.0, 1.5, 0.0, 0.8]
    for n in range(1, N):
        x_true[n] = F @ x_true[n - 1] + rng.multivariate_normal(np.zeros(4), Q)

    # Filtre : initialise sur la premiere observation, vitesse inconnue
    filt = KalmanFilter(
        sigma_Q=sigma_Q,
        sigma_px=sigma_gps,
        sigma_py=sigma_gps,
        x_init=[x_true[0, 0], 0.0, x_true[0, 2], 0.0],
        P_init=np.diag([sigma_gps**2, 100.0, sigma_gps**2, 100.0]),
    )

    estimated, gps_log = [], []
    for n in range(N):
        if blackout[0] <= n < blackout[1]:
            y = None  # zone sans GPS
            gps_log.append([np.nan, np.nan])
        else:
            y = H @ x_true[n] + rng.multivariate_normal(np.zeros(2), R)
            gps_log.append(y)
        x_est = filt.kallman_iter(y)
        estimated.append([x_est[0], x_est[2]])

    true_pos = x_true[:, [0, 2]]
    gps_arr = np.asarray(gps_log)
    mask = ~np.isnan(gps_arr[:, 0])

    mse_filt = calc_mse(estimated, true_pos)
    mse_gps = calc_mse(gps_arr[mask], true_pos[mask])
    err = np.asarray(estimated) - true_pos

    print(f"Scenario : N={N}, blackout steps {blackout[0]} a {blackout[1]}")
    print(f"MSE filtre  (toutes les steps)   : {mse_filt:.2f}")
    print(f"MSE GPS brut (hors blackout)      : {mse_gps:.2f}")
    print(
        f"Erreur euclidienne moyenne filtre : "
        f"{np.mean(np.linalg.norm(err, axis=1)):.2f} px"
    )
    return mse_filt


if __name__ == "__main__":
    offline_test()
