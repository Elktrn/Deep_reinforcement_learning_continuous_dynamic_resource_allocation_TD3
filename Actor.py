class Actor(nn.Module):
	def __init__(self, state_dim, action_dim, max_action):
		super(Actor, self).__init__()
		self.l1 = nn.Linear(state_dim, 128)
		self.l2 = nn.Linear(128, 128)
		self.l3 = nn.Linear(128, action_dim)
		self.max_action = max_action
	def forward(self, state):
		a = F.relu(self.l1(state))
		a = F.relu(self.l2(a))
		return self.max_action * torch.sigmoid(self.l3(a))
