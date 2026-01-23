"""
Performance metrics and profiling for the pipeline
"""
import time
import functools
import logging
from typing import Callable, Any
import pandas as pd


class PerformanceTracker:
    """Track performance metrics for pipeline operations"""
    
    def __init__(self):
        self.metrics = {}
        self.logger = logging.getLogger(__name__)
    
    def time_function(self, func: Callable) -> Callable:
        """Decorator to time function execution"""
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            start_time = time.time()
            result = func(*args, **kwargs)
            end_time = time.time()
            
            execution_time = end_time - start_time
            func_name = func.__name__
            
            if func_name not in self.metrics:
                self.metrics[func_name] = []
            
            self.metrics[func_name].append(execution_time)
            self.logger.info(f"{func_name} took {execution_time:.2f}s")
            
            return result
        return wrapper
    
    def get_summary(self) -> pd.DataFrame:
        """Get summary of all tracked metrics"""
        summary_data = []
        
        for func_name, times in self.metrics.items():
            summary_data.append({
                'function': func_name,
                'calls': len(times),
                'total_time': sum(times),
                'avg_time': sum(times) / len(times),
                'min_time': min(times),
                'max_time': max(times)
            })
        
        return pd.DataFrame(summary_data).sort_values('total_time', ascending=False)
    
    def print_report(self):
        """Print performance report"""
        print("\n" + "="*70)
        print("PERFORMANCE REPORT")
        print("="*70)
        
        df = self.get_summary()
        if not df.empty:
            for _, row in df.iterrows():
                print(f"\n{row['function']}:")
                print(f"  Calls: {row['calls']}")
                print(f"  Total Time: {row['total_time']:.2f}s")
                print(f"  Avg Time: {row['avg_time']:.2f}s")
                print(f"  Min/Max: {row['min_time']:.2f}s / {row['max_time']:.2f}s")
        
        print("\n" + "="*70)


# Global tracker instance
tracker = PerformanceTracker()
