from dataclasses import dataclass


@dataclass
class BudgetExceeded(Exception):
    
    def __init__(self, budget, spend = 0):
        
        self.budget = budget
        self.spend = spend
    def add(self, amount):
        
        self.spend += amount
    def check(self):
        
        if self.spend >= self.budget:
            raise BudgetExceeded(self.budget, self.spend)