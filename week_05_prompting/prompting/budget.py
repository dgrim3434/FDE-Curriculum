class BudgetExceeded(Exception):
    
    def __init__(self, budget: float, spend: float):
        
        super().__init__(f"Budget ${budget:.4f} reached (spend ${spend:.4f})")
        self.budget = budget
        self.spend = spend

class BudgetTracker:
    
    def __init__(self, budget: float):
        
        if budget < 0:
            raise ValueError(f"Budget must be >= 0, got {budget}")
        
        self.budget = budget
        self.spend = 0.0
    
    def add(self, amount: float) -> None:
        self.spend += amount
    
    @property
    def remaining(self) -> float:
        return self.budget - self.spend
    
    def check(self) -> None:
        
        if self.spend >= self.budget:
            raise BudgetExceeded(self.budget, self.spend)