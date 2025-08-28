from datetime import datetime, timezone
from decimal import Decimal

from src.core.config import Config
from src.exchanges.binance_websocket import BinanceWebSocketHandler


def test_binance_websocket_message_conversions():
    dummy_client = type("C", (), {"tld": "com", "testnet": False})()
    ws = BinanceWebSocketHandler(Config(), client=dummy_client, exchange_name="binance")

    # Ticker message
    msg = {
        "s": "BTCUSDT",
        "b": "49999.00",
        "a": "50001.00",
        "c": "50000.00",
        "v": "1000.0",
        "p": "100.00",
        "E": int(datetime.now(timezone.utc).timestamp() * 1000),
    }
    t = ws._convert_ticker_message(msg)
    assert t.symbol == "BTCUSDT"
    assert t.last_price == Decimal("50000.00")

    # Order book message
    ob = {
        "s": "BTCUSDT",
        "b": [["49999.00", "1.0"]],
        "a": [["50001.00", "1.0"]],
        "E": int(datetime.now(timezone.utc).timestamp() * 1000),
    }
    book = ws._convert_orderbook_message(ob)
    assert book.symbol == "BTCUSDT"
    assert book.bids[0][0] == Decimal("49999.00")

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

