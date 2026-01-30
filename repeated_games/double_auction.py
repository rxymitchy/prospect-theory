# Double Auction Game
class DoubleAuction:
  def __init__(self, k=10, valuation=6, cost=4):
    self.A = set(range(1, k+1))
    self.B = set(range(1, k+1))
    self.k = k

    self.valuation = valuation
    self.cost = cost

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
