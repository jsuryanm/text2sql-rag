import random 

from datetime import datetime, timedelta
from faker import Faker # generates fake data

import psycopg2
from psycopg2.extras import execute_values
# we can insert multiple rows directly without using for loop
import os 
from dotenv import load_dotenv

'''
Take this collection of Python tuples and efficiently put them into the VALUES section of my SQL query
from psycopg2.extras import execute_values

execute_values(
    cursor,
    """
    INSERT INTO users (name, email)
    VALUES %s
    """,
    users
)

INSERT INTO users (name, email)
VALUES
    ('Surya', 'surya@gmail.com'),
    ('Rahul', 'rahul@gmail.com'),
    ('Priya', 'priya@gmail.com');

'''

load_dotenv()

faker = Faker()

DATABASE_URL = os.getenv("DATABASE_URL")

if not DATABASE_URL:
    print("ERROR: DATABASE_URL not found in env")
    print("Please set the DATABASE_URL in .env")
    exit(1)

def generate_customers(count=100):
    """Generate customer data."""
    customers = []
    segments = ['SMB', 'Enterprise', 'Individual']
    countries = ['USA', 'UK', 'Canada', 'Germany', 'France', 'Australia', 'India', 'Singapore']

    for _ in range(count):
        customer = (
            faker.name(),
            faker.email(),
            random.choice(segments),
            random.choice(countries)
        )
        customers.append(customer)

    return customers    

def generate_products(count=50):
    """Generate product data."""
    products = []
    categories = ['Electronics', 'Software', 'Hardware', 'Services', 'Accessories', 'Books']

    product_names = {
        'Electronics': ['Laptop', 'Monitor', 'Keyboard', 'Mouse', 'Webcam', 'Headphones', 'Speaker'],
        'Software': ['Project Management Tool', 'Analytics Platform', 'CRM System', 'IDE License'],
        'Hardware': ['Server', 'Router', 'Switch', 'Cable', 'Adapter', 'Storage Drive'],
        'Services': ['Consulting', 'Training', 'Support Plan', 'Maintenance'],
        'Accessories': ['Case', 'Bag', 'Stand', 'Mount', 'Charger'],
        'Books': ['Programming Guide', 'Technical Manual', 'Design Book']
    }

    for i in range(count):
        category = random.choice(categories)
        base_name = random.choice(product_names.get(category, ['Product']))
        name = f"{base_name} {faker.company()}" if random.random() > 0.5 else base_name

        product = (
            name[:100],  # Limit to 100 chars
            category,
            round(random.uniform(10, 2000), 2),  # Price between $10 and $2000
            random.randint(0, 500),  # Stock quantity
            faker.text(max_nb_chars=200) if random.random() > 0.3 else None  # Description
        )
        products.append(product)

    return products

def generate_orders(customer_ids, count=200):
    """Generate order data."""
    orders = []
    statuses = ['Pending', 'Delivered', 'Cancelled', 'Processing']
    status_weights = [0.1, 0.7, 0.1, 0.1]  # Most orders are delivered

    # Generate orders from the past year
    end_date = datetime.now()
    start_date = end_date - timedelta(days=365)

    for _ in range(count):
        order_date = faker.date_between(start_date=start_date, end_date=end_date)

        order = (
            random.choice(customer_ids),
            order_date,
            round(random.uniform(50, 5000), 2),  # Order amount between $50 and $5000
            random.choices(statuses, weights=status_weights)[0],
            faker.address() if random.random() > 0.2 else None
        )
        orders.append(order)

    return orders

def main():
    """Main data generation function"""
    print("="*60)
    print("Multi-Source RAG + Text-to-SQL Sample Data Generator")
    print("="*60)

    try:
        print("\nConnecting to database")
        conn = psycopg2.connect(DATABASE_URL)
        cur = conn.cursor()
        print("Connected successfully")

        print("\nGenerating customers")
        customers = generate_customers(100)

        execute_values(
            cur,
            "INSERT INTO customers (name, email, segment, country) VALUES %s RETURNING id", # returning means after inserting row, postgres will return the id it generated    
            customers
        )

        customer_ids = [row[0] for row in cur.fetchall()]
        print(f"Inserted {len(customer_ids)} into customers")

        
        products = generate_products(50)
        
        execute_values(
            cur,
            "INSERT INTO products (name, category, price, stock_quantity, description) VALUES %s", # returning means after inserting row, postgres will return the id it generated    
            products
        )

        print(f"Inserted {len(products)} products")
        


        print(f"\nGenerating the orders..")
        
        orders = generate_orders(customer_ids, 200)
        execute_values(
            cur,
            "INSERT INTO orders (customer_id, order_date, total_amount, status, shipping_address) VALUES %s",
            orders
        )
        print(f"Inserted {len(orders)} orders")

        conn.commit() # saves all the changes made in current current transaction
        print("All data committed successfully")

        cur.execute("SELECT COUNT(*) FROM customers") 
        # returns the total no of rows, including the duplicates
        print(f"Customers: {cur.fetchone()[0]}")

        cur.execute("SELECT COUNT(*) FROM products") 
        print(f"Products: {cur.fetchone()[0]}")

        cur.execute("SELECT COUNT(*) FROM orders") 
        print(f"Orders: {cur.fetchone()[0]}")
        
        cur.execute("SELECT SUM(total_amount) FROM orders")
        total_revenue = cur.fetchone()[0]
        print(f"Total Revenue: ${total_revenue:,.2f}")

        cur.execute("SELECT COUNT(*) FROM orders WHERE status = 'Delivered'")
        delivered = cur.fetchone()[0]
        print(f"Delivered Orders: {delivered}")

        print("=" * 60)
        print("\n✓ Sample data generation complete!")
        print("\nYou can now use the Text-to-SQL features with this data.")

        # Close connection
        cur.close()
        conn.close()

    except psycopg2.Error as e:
        print(f"\n✗ Database error: {e}")
        print("\nMake sure:")
        print("1. DATABASE_URL is correct in your .env file")
        print("2. The database schema has been created (run data/sql/schema.sql)")
        print("3. Your database user has INSERT permissions")
        exit(1)

    except Exception as e:
        print(f"\n✗ Error: {e}")
        exit(1)


if __name__ == "__main__":
    main()
        


