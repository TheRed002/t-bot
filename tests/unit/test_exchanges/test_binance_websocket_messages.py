from datetime import datetime, timezone
from decimal import Decimal

from src.core.config import Config
from src.core.types import OrderSide
from src.exchanges.binance_websocket import BinanceWebSocketHandler


def test_binance_websocket_message_conversions():
    dummy_client = type("C", (), {"tld": "com", "testnet": False})()
    ws = BinanceWebSocketHandler(Config(), client=dummy_client, exchange_name="binance")

    # Ticker message - more complete Binance ticker format
    msg = {
        "s": "BTCUSDT",  # symbol
        "b": "49999.00",  # best bid price
        "B": "10.0",  # best bid quantity
        "a": "50001.00",  # best ask price
        "A": "5.0",  # best ask quantity
        "c": "50000.00",  # last price
        "Q": "0.1",  # last quantity
        "o": "49900.00",  # open price
        "h": "50100.00",  # high price
        "l": "49800.00",  # low price
        "v": "1000.0",  # 24h volume
        "q": "50000000.0",  # 24h quote volume
        "p": "100.00",  # 24h price change
        "P": "0.20",  # 24h price change percent
        "E": int(datetime.now(timezone.utc).timestamp() * 1000),
    }
    t = ws._convert_ticker_message(msg)
    assert t.symbol == "BTCUSDT"
    assert t.last_price == Decimal("50000.00")
    assert t.bid_price == Decimal("49999.00")
    assert t.ask_price == Decimal("50001.00")
    assert t.volume == Decimal("1000.0")
    assert t.exchange == "binance"

    # Order book message
    ob = {
        "s": "BTCUSDT",
        "b": [["49999.00", "1.0"]],
        "a": [["50001.00", "1.0"]],
        "E": int(datetime.now(timezone.utc).timestamp() * 1000),
    }
    book = ws._convert_orderbook_message(ob)
    assert book.symbol == "BTCUSDT"
    assert book.bids[0].price == Decimal("49999.00")
    assert book.asks[0].price == Decimal("50001.00")
    assert book.exchange == "binance"

    # Trade message
    trade = {
        "t": 1,
        "s": "BTCUSDT",
        "m": True,
        "q": "0.1",
        "p": "50000.00",
        "T": int(datetime.now(timezone.utc).timestamp() * 1000),
    }
    tr = ws._convert_trade_message(trade)
    assert tr.symbol == "BTCUSDT"
    assert tr.price == Decimal("50000.00")
    assert tr.quantity == Decimal("0.1")
    assert tr.exchange == "binance"
    assert (
        tr.side == OrderSide.SELL.value
    )  # m=True means buyer is maker, so we record the trade as a sell
