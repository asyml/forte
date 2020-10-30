import os
import numpy as np
class Extractor_Cache:
  def __init__(self):
    pass

  def get_ID(self, data_pack, instance):
    key = str(pack.pack_id()) + "_" + str(instance.tid)
    return key

  def contains_datapack(self, data_pack, instance):
    key = get_ID(data_pack, instance)
    return os.path.exists(key + ".npy")

  def cache_datapack(self, data_pack, instance, tensor):
    key = get_ID(data_pack, instance)
    np.save(key+".npy", tensor)
  
  #call contains_datapack to check cache firstly
  def get_tensor(self, data_pack, instance):
    key = get_ID(data_pack, instance)
    return np.load(key+".npy")