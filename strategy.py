import backtrader as bt

# now we develop a portfolio trading strategy


class TestStrategy(bt.Strategy):
    params = (
        ("hold_days", 10),
    )

    def __init__(self):
        # Keep a reference to the "close" line in the data[0] dataseries
        self.hold_counters = {d._name: 0 for d in self.datas}
        self.in_positions = {d._name: False for d in self.datas}

    def next(self):
        for i, d in enumerate(self.datas):
            if self.in_positions[d._name]:
                if self.hold_counters[d._name] >= self.params.hold_days:
                    self.sell(data=d, size=20)
                    self.in_positions[d._name] = False
                    self.hold_counters[d._name] = 0
                else:
                    self.hold_counters[d._name] += 1

            elif d.signal[0] == 1:
                self.buy(data=d, size=20)
                self.in_positions[d._name] = True

    def notify_order(self, order):
        # 未被处理的订单
        if order.status in [order.Submitted, order.Accepted]:
            return
        # 已经处理的订单
        if order.status in [order.Completed, order.Canceled, order.Margin]:
            if order.isbuy():
                self.log(
                    "Order Reference: %.0f, Executed Price: %.2f, Executed Amount: %.2f, Commission: %.2f, Executed Volume: %.2f, Stock Name: %s"
                    % (
                        order.ref,
                        order.executed.price,
                        order.executed.price * abs(order.executed.size),
                        order.executed.comm,
                        order.executed.size,
                        order.data._name,
                        )
                    )  # 股票名称
            else:  # Sell
                self.log(
                    "Order Reference: %.0f, Executed Price: %.2f, Executed Amount: %.2f, Commission: %.2f, Executed Volume: %.2f, Stock Name: %s"
                    % (
                        order.ref,
                        order.executed.price,
                        order.executed.price * abs(order.executed.size),
                        order.executed.comm,
                        order.executed.size,
                        order.data._name,
                        )
                    )

    def log(self, txt):
        dt = str(self.datas[0].datetime.datetime())
        print("%s, %s" % (dt, txt))

    def stop(self):
        self.log(
            "The Excessive Return is {}.".format(self.broker.getvalue() / 100_000 - 1 - 0.03),
        )
