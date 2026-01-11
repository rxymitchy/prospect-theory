# Double Auction Game
class DoubleAuction:
  def __init__(self, k=100):
    self.A = set(range(k+1))
    self.B = set(range(k+1))
    self.k = k

    # Static, arbitrary valuation and costs for now.
    # I'll spend some time figuring out how to handle this
    self.valuation = 65
    self.cost = 45

  def reset(self):
    self.valuation = np.random.randint(0, 101)
    self.cost = np.random.randint(0, 101)

  def step(self, bid, ask):

    assert bid in self.B and ask in self.A

    # If a successful trade
    if bid >= ask:
      price = (bid + ask) / 2
      buyer_payoff = self.valuation - price
      seller_payoff = price - self.cost
      return buyer_payoff, seller_payoff

    else:
      return 0, 0
