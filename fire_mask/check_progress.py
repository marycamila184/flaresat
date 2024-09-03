import pandas as pd

df = pd.read_csv('/home/marycamila/flaresat/source/landsat_scenes/2019/scenes_09_queue.csv')

print("Processed: " + str(len(df[df['fire_processed']])))
print("Queued: " + str(len(df[~df['fire_processed']])))