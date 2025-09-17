pip install backtrader yfinance
import backtrader as bt
import yfinance as yf

# Download historical data
data = yf.download("AAPL", start="2020-01-01", end="2023-01-01")

# Convert to Backtrader feed
data_feed = bt.feeds.PandasData(dataname=data)
class MovingAverageCrossStrategy(bt.Strategy):
    def __init__(self):
        self.sma_short = bt.indicators.SMA(period=50)
        self.sma_long = bt.indicators.SMA(period=200)

    def next(self):
        if not self.position:  # no open trade
            if self.sma_short[0] > self.sma_long[0]:
                self.buy(size=10)  # buy 10 shares
        else:
            if self.sma_short[0] < self.sma_long[0]:
                self.close()  # exit position


cerebro = bt.Cerebro()
cerebro.addstrategy(MovingAverageCrossStrategy)
cerebro.adddata(data_feed)
cerebro.broker.setcash(10000)  # starting capital
cerebro.broker.setcommission(commission=0.001)  # trading fee

print("Starting Portfolio Value:", cerebro.broker.getvalue())
cerebro.run()
print("Final Portfolio Value:", cerebro.broker.getvalue())
cerebro.plot()