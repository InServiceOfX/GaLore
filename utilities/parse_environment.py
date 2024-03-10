import os

class ParseEnvironment:
	def __init__(self):
		assert "LOCAL_RANK" in os.environ, "torchrun should set LOCAL_RANK"
		self.global_rank = int(os.environ['RANK'])
		self.local_rank = int(os.environ['LOCAL_RANK'])
		self.world_size = int(os.environ["WORLD_SIZE"])