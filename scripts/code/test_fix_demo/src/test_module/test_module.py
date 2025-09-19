#!/usr/bin/env python3
"""
Test module with various pylint errors for testing the pylint-fixer
This file intentionally contains errors to test the fixing capability
"""

from typing import List, Dict, Optional
import json

# E0602: Undefined variable (using variable before assignment)
def process_data(data):
    if len(data) > 0:
        result = processed_value * 2  # processed_value is undefined
    return result  # result might be undefined

# E1101: No member (accessing non-existent attribute)
class DataProcessor:
    def __init__(self):
        self.data = []
    
    def process(self):
        return self.non_existent_method()  # This method doesn't exist
    
    def analyze(self):
        config = {"key": "value"}
        return config.non_existent_key  # Dict doesn't have this attribute

# E1136: Unsubscriptable object
def get_item():
    value = None
    return value[0]  # None is not subscriptable

# E0611: No name in module (incorrect import)
from os import non_existent_function  # This doesn't exist in os

# E1120: No value for parameter
def function_with_params(a, b, c):
    return a + b + c

def call_function():
    return function_with_params(1, 2)  # Missing parameter c

# E1123: Unexpected keyword argument
def simple_function(x, y):
    return x + y

def call_with_wrong_args():
    return simple_function(x=1, y=2, z=3)  # z is unexpected

# E0601: Used before assignment
def conditional_assignment(flag):
    if flag:
        var = 10
    return var  # var might not be defined

# E1130: Invalid unary operand type
def invalid_operation():
    value = None
    return -value  # Can't use unary minus on None

# Complex class with multiple issues
class TradingBot:
    def __init__(self):
        self.balance = 1000.0  # Should use Decimal for money
        
    def place_order(self, amount):
        # E1101: Instance has no member
        self.orders.append(amount)  # orders doesn't exist
        
        # E0602: Undefined variable
        total = current_balance + amount  # current_balance undefined
        
        # E1136: Unsubscriptable
        settings = self.get_settings()
        return settings[0]  # What if settings is None?
    
    def get_settings(self):
        # Might return None
        return None
    
    def calculate_profit(self):
        # E1101: No member
        return self.profit_margin * 100  # profit_margin doesn't exist

# Function with type hint issues
def process_list(items: List[str]) -> Dict:
    # E0602: Undefined variable
    for item in items:
        processed_items.append(item.upper())  # processed_items undefined
    
    # E0601: Used before assignment
    if len(items) > 5:
        result_dict = {"status": "ok"}
    
    return result_dict  # Might not be defined

# Missing imports that are used
def save_data(data):
    # E0602: Undefined variable (should import Path)
    file_path = Path("data.json")  # Path not imported
    
    with open(file_path, 'w') as f:
        json.dump(data, f)

# Circular reference issue
class Node:
    def __init__(self, value):
        self.value = value
        self.next = None
    
    def get_chain_length(self):
        # E1101: Might cause issues if next doesn't have length
        return self.next.length()  # length() doesn't exist

# Function calling non-existent module function
def use_math():
    # E0611: No name in module
    from math import non_existent_math_func
    return non_existent_math_func(42)

# Class method with wrong self usage
class Calculator:
    @staticmethod
    def add(x, y):
        # E0602: self is not defined in static method
        return self.process(x + y)
    
    def process(self, value):
        return value * 2

# Type checking will fail
def type_confusion():
    values = "string"
    # E1136: String is not subscriptable like a list
    for i in range(10):
        print(values[i])  # Will fail if string is shorter

if __name__ == "__main__":
    # Test the functions with errors
    processor = DataProcessor()
    bot = TradingBot()
    
    # These will cause runtime errors
    # processor.process()
    # bot.place_order(100)
    print("Module with errors loaded")