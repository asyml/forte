class Extractor_Cache:
  #modify test
  def __init__(self):
    self.cache_datapack2tensor = dict()

  def contains_datapack(self, data_pack):
    if data_pack in self.cache_datapack2tensor:
      return True
    else:
      return False

  def cache_datapack(self, data_pack, tensor):
    cache_datapack2tensor[data_pack] = tensor
    

  def get_tensor(self, data_pack):
    return self.cache_datapack2tensor[data_pack]

  def remove_datapack(self, data_pack):
    del self.cache_datapack2tensor[data_pack]


  def clear_cache_datapack2tensor (self):
    self.cache_datapack2tensor.clear()


def gen():
  for pack in packs
    for instance in pack
      key = str(pack.pack_id()) + "_" + str(instance.tid)

      if cache.contains_datapack(key):
        yield cache.get_tensor(key)

      tensor = extract(instance)
      
      cache.cache_datapack(key,tensor)

      yield tensor



if __name__ == '__main__':
  cache = Converter_Cache()

  for tensor in gen():
    print(tensor)
