# NB: constants for testing, might want to move these elsewhere 
# for cleanless

### Map exploration constants ###
FRONTIER_UPDATE_INTERVAL = 100 # whatever happens, frontier is updated every x ticks; which causes recalculation of exploration waypoint as well
ARRIVAL_THRESHOLD = 50 # distance threshold to frontier exploration waypoint
MINIMUM_CLUSTER_SIZE = 50 # minimum cluster size to define a frontier
MINIMUM_DISTANCE = MINIMUM_CLUSTER_SIZE + 0 # minimum distance to drone for our waypoint candidates for frontier exploration. NB needs to be bigger than arrival threshold
