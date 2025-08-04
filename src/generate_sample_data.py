import random
import pandas as pd

def generate_fake_data(num_rows=100):
    data = []
    device_types = ["mobile", "desktop", "tablet"]
    product_categories = ["electronics", "books", "clothing"]
    
    for i in range(num_rows):
        device = random.choice(device_types)
        category = random.choice(product_categories)
        time = round(random.uniform(2.0, 15.0), 1)
        clicked = 1 if time > 6 else 0
        data.append([i, device, category, time, clicked])
    
    df = pd.DataFrame(data, columns=["user_id", "device_type", "product_category", "time_on_page", "clicked"])
    return df

# Kullanımı:
df = generate_fake_data(200)
df.to_csv("data/sample_user_clicks.csv", index=False)
