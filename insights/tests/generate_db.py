import sqlite3
import random
from datetime import datetime, timedelta
import json

# Define constants
START_DATE = datetime(2025, 1, 1)
END_DATE = datetime(2025, 4, 13)
PRICE_INCREASE_DATE = datetime(2025, 3, 1)
PRICE_INCREASE_PERCENTAGE = 0.08  # 8% increase

# Laptop brands and models
laptop_data = {
    "Dell": [
        {"model_name": "XPS 13", "processor": "Intel Core i7", "ram_gb": 16, "storage_gb": 512, "display_size": 13.3, "base_price": 1499.99},
        {"model_name": "Inspiron 15", "processor": "Intel Core i5", "ram_gb": 8, "storage_gb": 256, "display_size": 15.6, "base_price": 899.99},
        {"model_name": "Precision 5570", "processor": "Intel Core i9", "ram_gb": 32, "storage_gb": 1024, "display_size": 15.6, "base_price": 2299.99}
    ],
    "HP": [
        {"model_name": "Spectre x360", "processor": "Intel Core i7", "ram_gb": 16, "storage_gb": 512, "display_size": 13.5, "base_price": 1599.99},
        {"model_name": "Pavilion 15", "processor": "AMD Ryzen 7", "ram_gb": 16, "storage_gb": 512, "display_size": 15.6, "base_price": 999.99},
        {"model_name": "EliteBook 840", "processor": "Intel Core i5", "ram_gb": 8, "storage_gb": 256, "display_size": 14.0, "base_price": 1299.99}
    ],
    "Lenovo": [
        {"model_name": "ThinkPad X1", "processor": "Intel Core i7", "ram_gb": 16, "storage_gb": 512, "display_size": 14.0, "base_price": 1799.99},
        {"model_name": "Yoga 9i", "processor": "Intel Core i7", "ram_gb": 16, "storage_gb": 512, "display_size": 14.0, "base_price": 1499.99},
        {"model_name": "IdeaPad 5", "processor": "AMD Ryzen 5", "ram_gb": 8, "storage_gb": 256, "display_size": 15.6, "base_price": 849.99}
    ],
    "Apple": [
        {"model_name": "MacBook Pro 14", "processor": "Apple M2 Pro", "ram_gb": 16, "storage_gb": 512, "display_size": 14.2, "base_price": 1999.99},
        {"model_name": "MacBook Air", "processor": "Apple M2", "ram_gb": 8, "storage_gb": 256, "display_size": 13.6, "base_price": 1199.99},
        {"model_name": "MacBook Pro 16", "processor": "Apple M2 Max", "ram_gb": 32, "storage_gb": 1024, "display_size": 16.2, "base_price": 2699.99}
    ],
    "Asus": [
        {"model_name": "ZenBook 14", "processor": "Intel Core i7", "ram_gb": 16, "storage_gb": 512, "display_size": 14.0, "base_price": 1299.99},
        {"model_name": "ROG Zephyrus", "processor": "AMD Ryzen 9", "ram_gb": 16, "storage_gb": 1024, "display_size": 14.0, "base_price": 1799.99},
        {"model_name": "VivoBook 15", "processor": "Intel Core i5", "ram_gb": 8, "storage_gb": 512, "display_size": 15.6, "base_price": 799.99}
    ]
}

# Accessories
accessories = [
    {"name": "Wireless Mouse", "type": "Mouse", "base_price": 49.99},
    {"name": "Bluetooth Keyboard", "type": "Keyboard", "base_price": 79.99},
    {"name": "Noise-Cancelling Headphones", "type": "Headset", "base_price": 129.99},
    {"name": "27-inch Monitor", "type": "Monitor", "base_price": 349.99},
    {"name": "USB Hub", "type": "USB Hub", "base_price": 39.99},
    {"name": "Laptop Docking Station", "type": "Docking Station", "base_price": 149.99},
    {"name": "Laptop Backpack", "type": "Laptop Bag", "base_price": 79.99},
    {"name": "HD Webcam", "type": "Webcam", "base_price": 59.99}
]

def create_database():
    # Connect to SQLite database (creates a new file if it doesn't exist)
    conn = sqlite3.connect('it_sales.db')
    cursor = conn.cursor()
    
    # Enable foreign keys
    cursor.execute('PRAGMA foreign_keys = ON')
    
    # Create tables
    create_tables(cursor)
    
    # Generate data
    products = []
    laptop_models = []
    accessory_data = []
    price_history = []
    
    # Add laptop products
    product_id = 1
    laptop_id = 1
    
    for brand, models in laptop_data.items():
        for model in models:
            # Base product data
            product_name = f"{brand} {model['model_name']}"
            description = f"{model['processor']}, {model['ram_gb']}GB RAM, {model['storage_gb']}GB SSD, {model['display_size']}\" Display"
            
            products.append((
                product_id,
                product_name,
                "Laptop",
                description,
                model['base_price'],
                model['base_price'],  # Current price same as base initially
                random.randint(10, 50),  # Random stock
                START_DATE - timedelta(days=random.randint(30, 90)),  # Created before start date
                START_DATE - timedelta(days=random.randint(1, 29))    # Updated before start date
            ))
            
            # Laptop model data
            laptop_models.append((
                laptop_id,
                product_id,
                brand,
                model['model_name'],
                model['processor'],
                model['ram_gb'],
                model['storage_gb'],
                model['display_size'],
                "Integrated Graphics" if random.random() < 0.6 else "Dedicated Graphics",
                random.choice([0, 1]),  # Random touchscreen (0=no, 1=yes)
                "Windows 11" if brand != "Apple" else "macOS"
            ))
            
            # Initial price history
            created_date = products[-1][7]
            price_history.append((
                len(price_history) + 1,
                product_id,
                created_date,
                model['base_price'],
                "Initial Price"
            ))
            
            # March 1st price increase
            new_price = round(model['base_price'] * (1 + PRICE_INCREASE_PERCENTAGE), 2)
            price_history.append((
                len(price_history) + 1,
                product_id,
                PRICE_INCREASE_DATE,
                new_price,
                "March 1st Price Increase"
            ))
            
            laptop_id += 1
            product_id += 1
    
    # Add accessory products
    accessory_id = 1
    for accessory in accessories:
        # Base product data
        product_name = accessory['name']
        description = f"{accessory['type']} for laptops"
        
        products.append((
            product_id,
            product_name,
            "Accessory",
            description,
            accessory['base_price'],
            accessory['base_price'],  # Current price same as base initially
            random.randint(20, 100),  # Random stock
            START_DATE - timedelta(days=random.randint(30, 90)),  # Created before start date
            START_DATE - timedelta(days=random.randint(1, 29))    # Updated before start date
        ))
        
        # Accessory data
        accessory_data.append((
            accessory_id,
            product_id,
            accessory['type'],
            json.dumps([])  # Empty compatible brands means compatible with all
        ))
        
        # Initial price history
        created_date = products[-1][7]
        price_history.append((
            len(price_history) + 1,
            product_id,
            created_date,
            accessory['base_price'],
            "Initial Price"
        ))
        
        # March 1st price increase
        new_price = round(accessory['base_price'] * (1 + PRICE_INCREASE_PERCENTAGE), 2)
        price_history.append((
            len(price_history) + 1,
            product_id,
            PRICE_INCREASE_DATE,
            new_price,
            "March 1st Price Increase"
        ))
        
        accessory_id += 1
        product_id += 1
    
    # Generate customers, sales orders, and order items
    customers = generate_customers(200)
    sales_orders, order_items = generate_sales_data(products, customers)
    
    # Insert data into tables
    cursor.executemany("INSERT INTO products VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)", products)
    cursor.executemany("INSERT INTO laptop_models VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)", laptop_models)
    cursor.executemany("INSERT INTO accessories VALUES (?, ?, ?, ?)", accessory_data)
    cursor.executemany("INSERT INTO price_history VALUES (?, ?, ?, ?, ?)", price_history)
    cursor.executemany("INSERT INTO customers VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)", customers)
    cursor.executemany("INSERT INTO sales_orders VALUES (?, ?, ?, ?, ?, ?)", sales_orders)
    cursor.executemany("INSERT INTO order_items VALUES (?, ?, ?, ?, ?, ?)", order_items)
    
    # Create views
    create_views(cursor)
    
    # Commit changes and close connection
    conn.commit()
    conn.close()
    
    print(f"Database created successfully with:")
    print(f"- {len(products)} products ({len(laptop_models)} laptops, {len(accessory_data)} accessories)")
    print(f"- {len(customers)} customers")
    print(f"- {len(sales_orders)} sales orders")
    print(f"- {len(order_items)} order items")
    print(f"- {len(price_history)} price history records")
    
def create_tables(cursor):
    # Products table
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS products (
        product_id INTEGER PRIMARY KEY,
        name TEXT NOT NULL,
        category TEXT NOT NULL,
        description TEXT,
        base_price REAL NOT NULL,
        current_price REAL NOT NULL,
        stock_quantity INTEGER NOT NULL,
        created_at TIMESTAMP NOT NULL,
        updated_at TIMESTAMP NOT NULL
    )
    ''')
    
    # Laptop models table
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS laptop_models (
        model_id INTEGER PRIMARY KEY,
        product_id INTEGER NOT NULL,
        brand TEXT NOT NULL,
        model_name TEXT NOT NULL,
        processor TEXT NOT NULL,
        ram_gb INTEGER NOT NULL,
        storage_gb INTEGER NOT NULL,
        display_size REAL NOT NULL,
        graphics TEXT NOT NULL,
        is_touchscreen BOOLEAN NOT NULL,
        operating_system TEXT NOT NULL,
        FOREIGN KEY (product_id) REFERENCES products (product_id)
    )
    ''')
    
    # Accessories table
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS accessories (
        accessory_id INTEGER PRIMARY KEY,
        product_id INTEGER NOT NULL,
        type TEXT NOT NULL,
        compatible_brands TEXT,
        FOREIGN KEY (product_id) REFERENCES products (product_id)
    )
    ''')
    
    # Customers table
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS customers (
        customer_id INTEGER PRIMARY KEY,
        first_name TEXT NOT NULL,
        last_name TEXT NOT NULL,
        email TEXT NOT NULL,
        phone TEXT,
        address TEXT,
        company_name TEXT,
        is_business BOOLEAN NOT NULL,
        created_at TIMESTAMP NOT NULL
    )
    ''')
    
    # Sales orders table
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS sales_orders (
        order_id INTEGER PRIMARY KEY,
        customer_id INTEGER NOT NULL,
        order_date TIMESTAMP NOT NULL,
        total_amount REAL NOT NULL,
        payment_method TEXT NOT NULL,
        status TEXT NOT NULL,
        FOREIGN KEY (customer_id) REFERENCES customers (customer_id)
    )
    ''')
    
    # Order items table
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS order_items (
        order_item_id INTEGER PRIMARY KEY,
        order_id INTEGER NOT NULL,
        product_id INTEGER NOT NULL,
        quantity INTEGER NOT NULL,
        unit_price REAL NOT NULL,
        subtotal REAL NOT NULL,
        FOREIGN KEY (order_id) REFERENCES sales_orders (order_id),
        FOREIGN KEY (product_id) REFERENCES products (product_id)
    )
    ''')
    
    # Price history table
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS price_history (
        price_history_id INTEGER PRIMARY KEY,
        product_id INTEGER NOT NULL,
        effective_date TIMESTAMP NOT NULL,
        price REAL NOT NULL,
        reason TEXT,
        FOREIGN KEY (product_id) REFERENCES products (product_id)
    )
    ''')

def generate_customers(num_customers):
    """Generate customer data"""
    customers = []
    
    first_names = ["James", "Mary", "John", "Patricia", "Robert", "Jennifer", "Michael", "Linda", "William", "Elizabeth",
                  "David", "Susan", "Richard", "Jessica", "Joseph", "Sarah", "Thomas", "Karen", "Charles", "Nancy"]
    
    last_names = ["Smith", "Johnson", "Williams", "Jones", "Brown", "Davis", "Miller", "Wilson", "Moore", "Taylor",
                 "Anderson", "Thomas", "Jackson", "White", "Harris", "Martin", "Thompson", "Garcia", "Martinez", "Robinson"]
    
    companies = ["ABC Corp", "XYZ Inc", "Tech Solutions", "Digital Innovations", "Global Systems", "Nexus Technologies"]
    
    for customer_id in range(1, num_customers + 1):
        first_name = random.choice(first_names)
        last_name = random.choice(last_names)
        email = f"{first_name.lower()}.{last_name.lower()}@{'gmail' if random.random() < 0.7 else 'outlook'}.com"
        
        is_business = random.random() < 0.3  # 30% chance of being a business customer
        
        customers.append((
            customer_id,
            first_name,
            last_name,
            email,
            f"+1-{random.randint(200, 999)}-{random.randint(100, 999)}-{random.randint(1000, 9999)}",
            f"{random.randint(1, 9999)} {random.choice(['Main', 'Oak', 'Pine', 'Maple'])} {random.choice(['St', 'Ave', 'Blvd', 'Rd'])}",
            random.choice(companies) if is_business else None,
            1 if is_business else 0,  # SQLite stores booleans as 0/1
            START_DATE - timedelta(days=random.randint(1, 365))
        ))
    
    return customers

def generate_sales_data(products, customers):
    """Generate sales orders and order items"""
    sales_orders = []
    order_items = []
    
    order_id = 1
    order_item_id = 1
    
    # Payment methods and statuses
    payment_methods = ["Credit Card", "PayPal", "Bank Transfer", "Financing"]
    statuses = ["Completed", "Pending", "Shipped", "Cancelled"]
    status_weights = [0.85, 0.05, 0.08, 0.02]  # Completed orders more common
    
    # Generate random sales throughout the date range
    num_days = (END_DATE - START_DATE).days + 1
    
    # Weekly pattern weights - weekdays have more sales than weekends
    day_weights = [0.7, 1.0, 1.0, 1.0, 1.0, 0.8, 0.6]  # Mon-Sun
    
    # Monthly sales growth
    monthly_multipliers = {1: 1.0, 2: 1.2, 3: 1.5, 4: 1.3}  # Jan-Apr
    
    # Target total number of orders
    target_orders = 1000
    
    # Distribute orders across days
    orders_per_day = {}
    for day_offset in range(num_days):
        current_date = START_DATE + timedelta(days=day_offset)
        month = current_date.month
        weekday = current_date.weekday()
        
        # Calculate day weight based on weekday and month
        day_weight = day_weights[weekday] * monthly_multipliers[month]
        
        # Generate orders for this day
        orders_per_day[day_offset] = max(1, int(random.normalvariate(target_orders / num_days * day_weight, 2)))
    
    # Create orders for each day
    for day_offset, num_orders in orders_per_day.items():
        current_date = START_DATE + timedelta(days=day_offset)
        
        for _ in range(num_orders):
            # Select random customer
            customer = random.choice(customers)
            
            # Determine order status
            status = random.choices(statuses, weights=status_weights)[0]
            
            # Create order with random timestamp during business hours
            order_date = current_date.replace(
                hour=random.randint(8, 20),
                minute=random.randint(0, 59),
                second=random.randint(0, 59)
            )
            
            # Determine if this is a laptop purchase (30% chance) or accessory-only
            is_laptop_purchase = random.random() < 0.3
            
            # Select products and quantities
            order_products = []
            total_amount = 0
            
            # Filter products by category
            laptop_products = [p for p in products if p[2] == "Laptop"]
            accessory_products = [p for p in products if p[2] == "Accessory"]
            
            # Add laptop if this is a laptop purchase
            if is_laptop_purchase and laptop_products:
                laptop = random.choice(laptop_products)
                
                # Determine price based on order date (pre or post price increase)
                laptop_price = laptop[4]  # Base price
                if order_date >= PRICE_INCREASE_DATE:
                    laptop_price = round(laptop_price * (1 + PRICE_INCREASE_PERCENTAGE), 2)
                
                order_products.append({
                    "product": laptop,
                    "quantity": 1,
                    "unit_price": laptop_price
                })
                
                # 70% chance to add 1-3 accessories with a laptop
                if random.random() < 0.7 and accessory_products:
                    num_accessories = random.randint(1, 3)
                    selected_accessories = random.sample(accessory_products, min(num_accessories, len(accessory_products)))
                    
                    for accessory in selected_accessories:
                        # Determine price based on order date
                        accessory_price = accessory[4]  # Base price
                        if order_date >= PRICE_INCREASE_DATE:
                            accessory_price = round(accessory_price * (1 + PRICE_INCREASE_PERCENTAGE), 2)
                        
                        order_products.append({
                            "product": accessory,
                            "quantity": random.randint(1, 2),
                            "unit_price": accessory_price
                        })
            
            # If not a laptop purchase or no laptops available, add 1-3 accessories
            elif accessory_products:
                num_accessories = random.randint(1, 3)
                selected_accessories = random.sample(accessory_products, min(num_accessories, len(accessory_products)))
                
                for accessory in selected_accessories:
                    # Determine price based on order date
                    accessory_price = accessory[4]  # Base price
                    if order_date >= PRICE_INCREASE_DATE:
                        accessory_price = round(accessory_price * (1 + PRICE_INCREASE_PERCENTAGE), 2)
                    
                    order_products.append({
                        "product": accessory,
                        "quantity": random.randint(1, 3),
                        "unit_price": accessory_price
                    })
            
            # Calculate total amount and create order items
            for item in order_products:
                item_total = item["unit_price"] * item["quantity"]
                total_amount += item_total
            
            # Skip if no products were added
            if not order_products:
                continue
            
            # Add order
            sales_orders.append((
                order_id,
                customer[0],  # customer_id
                order_date,
                round(total_amount, 2),
                random.choice(payment_methods),
                status
            ))
            
            # Add order items
            for item in order_products:
                order_items.append((
                    order_item_id,
                    order_id,
                    item["product"][0],  # product_id
                    item["quantity"],
                    item["unit_price"],
                    round(item["unit_price"] * item["quantity"], 2)  # subtotal
                ))
                order_item_id += 1
            
            order_id += 1
    
    return sales_orders, order_items

def create_views(cursor):
    """Create useful database views"""
    # Monthly sales
    cursor.execute('''
    CREATE VIEW IF NOT EXISTS monthly_sales AS
    SELECT 
        strftime('%Y-%m', order_date) AS month,
        COUNT(DISTINCT order_id) AS number_of_orders,
        SUM(total_amount) AS total_revenue,
        AVG(total_amount) AS average_order_value
    FROM sales_orders
    WHERE status = 'Completed'
    GROUP BY strftime('%Y-%m', order_date)
    ORDER BY month
    ''')
    
    # Sales by product
    cursor.execute('''
    CREATE VIEW IF NOT EXISTS sales_by_product AS
    SELECT 
        p.product_id,
        p.name,
        p.category,
        COUNT(DISTINCT oi.order_id) AS number_of_orders,
        SUM(oi.quantity) AS total_quantity_sold,
        SUM(oi.subtotal) AS total_revenue
    FROM products p
    JOIN order_items oi ON p.product_id = oi.product_id
    JOIN sales_orders so ON oi.order_id = so.order_id
    WHERE so.status = 'Completed'
    GROUP BY p.product_id, p.name, p.category
    ''')
    
    # Sales by brand (for laptops)
    cursor.execute('''
    CREATE VIEW IF NOT EXISTS sales_by_brand AS
    SELECT 
        lm.brand,
        COUNT(DISTINCT oi.order_id) AS number_of_orders,
        SUM(oi.quantity) AS total_quantity_sold,
        SUM(oi.subtotal) AS total_revenue
    FROM laptop_models lm
    JOIN products p ON lm.product_id = p.product_id
    JOIN order_items oi ON p.product_id = oi.product_id
    JOIN sales_orders so ON oi.order_id = so.order_id
    WHERE so.status = 'Completed'
    GROUP BY lm.brand
    ORDER BY total_revenue DESC
    ''')

if __name__ == "__main__":
    # Delete the database if it already exists (for clean slate)
    import os
    if os.path.exists('it_sales.db'):
        os.remove('it_sales.db')
    
    create_database()
    print("SQLite database 'it_sales.db' created successfully!")