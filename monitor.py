# monitor.py
import requests
import time
import json
import pandas as pd
from datetime import datetime
import matplotlib.pyplot as plt

class PerformanceMonitor:
    def __init__(self, base_url="http://localhost:8000"):
        self.base_url = base_url
        self.metrics_data = []
    
    def collect_metrics(self, duration_minutes=5, interval_seconds=10):
        """Collect performance metrics over time"""
        print(f" Collecting metrics for {duration_minutes} minutes...")
        
        end_time = time.time() + (duration_minutes * 60)
        
        while time.time() < end_time:
            try:
                # Get health metrics
                health_response = requests.get(f"{self.base_url}/health", timeout=5)
                metrics_response = requests.get(f"{self.base_url}/metrics", timeout=5)
                
                if health_response.status_code == 200 and metrics_response.status_code == 200:
                    health_data = health_response.json()
                    metrics_data = metrics_response.json()
                    
                    timestamp = datetime.now().isoformat()
                    
                    self.metrics_data.append({
                        'timestamp': timestamp,
                        'cpu_percent': metrics_data['system']['cpu_percent'],
                        'memory_percent': metrics_data['system']['memory_percent'],
                        'uptime_seconds': health_data['uptime_seconds'],
                        'model_loaded': health_data['model_loaded']
                    })
                
                time.sleep(interval_seconds)
                
            except Exception as e:
                print(f"Error collecting metrics: {e}")
                time.sleep(interval_seconds)
    
    def generate_report(self, container_count):
        """Generate performance report"""
        if not self.metrics_data:
            print("No metrics data collected")
            return
        
        df = pd.DataFrame(self.metrics_data)
        
        # Create performance plots
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
        
        # CPU Usage
        ax1.plot(df['timestamp'], df['cpu_percent'])
        ax1.set_title(f'CPU Usage - {container_count} Containers')
        ax1.set_ylabel('CPU %')
        ax1.tick_params(axis='x', rotation=45)
        
        # Memory Usage
        ax2.plot(df['timestamp'], df['memory_percent'])
        ax2.set_title(f'Memory Usage - {container_count} Containers')
        ax2.set_ylabel('Memory %')
        ax2.tick_params(axis='x', rotation=45)
        
        # Uptime
        ax3.plot(df['timestamp'], df['uptime_seconds'])
        ax3.set_title('Service Uptime')
        ax3.set_ylabel('Seconds')
        ax3.tick_params(axis='x', rotation=45)
        
        # Summary stats
        summary_text = f"""
        Performance Summary - {container_count} Containers
        =================================
        Average CPU: {df['cpu_percent'].mean():.1f}%
        Max CPU: {df['cpu_percent'].max():.1f}%
        Average Memory: {df['memory_percent'].mean():.1f}%
        Samples: {len(df)}
        """
        
        ax4.text(0.1, 0.5, summary_text, fontsize=12, va='center')
        ax4.axis('off')
        
        plt.tight_layout()
        plt.savefig(f'performance_report_{container_count}_containers.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        print(f" Report generated: performance_report_{container_count}_containers.png")

if __name__ == "__main__":
    monitor = PerformanceMonitor()
    monitor.collect_metrics(duration_minutes=2)
    monitor.generate_report(container_count=1)