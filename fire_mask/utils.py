import pandas as pd

path_csv= '/home/marycamila/flaresat/source/landsat_scenes/2019/active_fire/scenes_12_queue.csv'

def check_progress():
    df = pd.read_csv(path_csv)
    print("Processed: " + str(len(df[df['fire_processed']])))
    print("Queued: " + str(len(df[~df['fire_processed']])))


def remove_entity(entity):
    df = pd.read_csv(path_csv)
    df = df[df["entity_id_sat"] != entity]
    df.to_csv(path_csv)


check_progress()

# Month 08 - Scene with errors
#remove_entity("LC81870282019229LGN00")
#remove_entity("LC81870322019213LGN00")
#remove_entity("LC81870322019229LGN00")
#remove_entity("LC81860282019222LGN00")
#remove_entity("LC81860292019222LGN00")
#remove_entity("LC81860322019222LGN00")
#remove_entity("LC81860322019238LGN00")
#remove_entity("LC81870272019229LGN00")
#remove_entity("LC81870182019229LGN00")
#remove_entity("LC81860622019222LGN00")
#remove_entity("LC81860492019222LGN00")

# Month 09 - Scene with saturation
#remove_entity("LC80420352019269LGN00")

