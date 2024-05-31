# Python Cheatsheet

### Functions

```py
def add(number):
	return number + 1
```

### F Strings

```py
print(f"My name is {name}")
```

### Classes

```py
class Person:
  def __init__(self, name, age):
    self.name = name
    self.age = age

  def greet(self):
    print("Hello, my name is " + self.name)

person1 = Person("John", 25)

person1.name
person1.greet()
```


### Iterating

```py
mylist = [1,2,3]

for value in mylist:
	print(value)
```

### Lists

```py
mylist = [1,2,3]

mylist.sort()
my_list.append(6)

# Remove 2 from list (only removes 1st element)]
my_list.remove(2) 

# Modify the last item in the list
my_list[-1] = 20 
```

### Tuples
Tuples are ordered and immutable (contents cant be changed)

```py
my_tuple = tuple([1, 2, 3])
my_tuple = (1, 2, 3)
```


### Dictionaries

```py
my_dict = {"key1": "value1", "key2": "value2"}
value1 = my_dict["key1"]
value1 = my_dict.get("key3", "default_value") #returns default_value
#add one dict to another
my_dict.update({"key4": "value4", "key5": "value5"}) 
del my_dict["key1"]
```


### Sets
* Unordered
* Unique Elements: duplicate entries are automatically removed.
* Mutable: You can add or remove elements from a set after its creation.
* Dynamic: A set's size changes as items are added or removed.
* Can determine if an item exists instantly in constant time (no iterating) 

```py
thisset = {"apple", "banana", "cherry"}
```

### Performance Comparison
#### Finding a member
* Lists and tuples - O(n)
* Sets  and Dictionaries - O(1)

### List comprehension

```py
numbers = [1,2,3,4,5]
even_squared_numbers = [n ** 2 for n in numbers if n % 2 == 0]
```

```py
numbers_1 = [1,2,3]
numbers_2 = [4,5,6]
pairs = [ (n1,n2) for n1 in numbers_1 for n2 in numbers_2 ]
```

### Lambda (anonymous) function
Here the lambda function is applied to number using map, we dont need to define a proper function…

```py
numbers = [1, 2, 3, 4, 5]
squared_numbers = list(map(lambda x: x**2, numbers))
```

### Error handling
The goal of error handling in automated trading is not just to prevent crashes but to ensure the system behaves predictably and responsibly. Given the high stakes in trading, it's essential to anticipate possible issues, handle them gracefully, and have robust logging and alerting in place to keep stakeholders informed.

Try-Except works like a try-catch block. Can catch general Exception to catch all other cases…
```py
try: 
	price = external_data_library.fetch_price() 
except 
	external_data_library.DataNotAvailableError: 
	# Handle the specific error, maybe by retrying, logging, or using backup data
	price = backup_data_source.fetch_price() 
except Exception as e: 
	# A catch-all for unexpected errors 
	log_error(f"Unexpected error: {e}") 
	stop_trading_and_alert()
```

Finally can be used to clean up…
```py
try: 
	start_trading() 
except (PriceDataError, ConnectionError) as e: 
	log_error(f"Critical error: {e}") 
	send_email_alert(f"Trading halted due to error: {e}") 
finally: 
	cleanup_resources() close_connections()
```

Retry mechanism:
```py
for _ in range(3): # Try 3 times 
	try: 
		price = fetch_price_from_server() 
		break # Exit the loop if successful 
	except ConnectionError: 
		wait_for_seconds(10) # Wait 10 seconds before retrying 
else: 
	# If we've exited the loop without breaking (i.e., all retries failed)
 	log_error("Failed to fetch price after 3 attempts.") 	
	use_backup_data_or_stop_trading()
```

### \_\_Name__ variable
Python assigns ”\_\_main__’ to the __name__ variable when you run the script directly and the module name if you import the script as a module. If we create billing.py :

```py
   def calculate_tax(price, tax):
    return price * tax


def print_billing_doc():
    tax_rate = 0.1
    products = [{'name': 'Book',  'price': 30},
                {'name': 'Pen', 'price': 5}]

    # print billing header
    print(f'Name\tPrice\tTax')

    # print the billing item
    for product in products:
        tax = calculate_tax(product['price'], tax_rate)
        print(f"{product['name']}\t{product['price']}\t{tax}")


print(__name__)
```

If the module is imported (`import billing`) the script executes and the result is “billing”. If we run the file directly (`python billing.py)`  the output is “__main__” 

Therefore, the __name__ variable allows you to check when the file is executed directly or imported as a module.

### **Multithreading and Multiprocessing**

Challenges:
1. Complexity - More too keep track of, more pitfalls
2. Synchronisation issues - Two threads may try to modify the same data at the same time
3. Deadlocks - Two threads may end up waiting for each other infinitely
4. Global Interpreter Lock (GIL) - Most common CPython implementation prevents multiple threads running concurrently in the same process 
5. Resource Limitations - Requires more memory and CPU => performance issues
6. Debugging issue - trickier to debug and find issues 

#### GIL (Global interpret lock)
GIL, is a mutex (short for mutual exclusion) that protects access to Python objects, preventing multiple native threads from executing Python bytecodes concurrently in a single process. In simpler terms, even if your Python code is running in a multi-threaded environment on a multi-core machine, only one thread can execute Python code at a time.

This makes it easy to :
1. Manage memory 
2. Use C Libraries which aren’t thread safe

Implications for Concurrent Code:
1. Doesn’t have much effect on I/O bound tasks - GIL can be released during I/O allowing other threads to run
2. CPU-Bound tasks - GIL is a bottleneck
3. Developers often use multiprocessing instead - creating different processes with their own memory space. They execute in parallel but IPC is more complex

#### Threading

```py
import threading 
import time 

def io_task(name, duration): 
	print(f"Thread {name} starting") 
	time.sleep(duration) 
	print(f"Thread {name} completed") # Create three threads that will run the `io_task` function 

thread1 = threading.Thread(target=io_task, args=("A", 2)) 
thread2 = threading.Thread(target=io_task, args=("B", 4)) 
thread3 = threading.Thread(target=io_task, args=("C", 1.5)) 

thread1.start() 
thread2.start() 
thread3.start() 

thread1.join()
thread2.join()
thread3.join()
print("All threads are done")
```

#### Parallelism with the multiprocessing Module

Multiprocessing bypasses the GIL by creating separate processes with individual memory space and Python interpreters.

```py
import multiprocessing 

def cpu_task(num): 
	result = 0 for i in range(num): 
	result += i**2 
	return result 

if __name__ == "__main__": 
	numbers = [10000000, 20000000, 30000000] 
	with multiprocessing.Pool() as pool: 
		results = pool.map(cpu_task, numbers) 
		print(results)
```

Pool creates n concurrent processes each running cpu_task(numers[n]).  The results are stored in results (a list) and printed. The With block is a context manager that ensures the pool is properly created and later cleaned up.

### **Asynchronous Programming with** async/await

This is another way to write concurrent code, particularly beneficial for I/O-bound tasks. It doesn't create separate threads or processes but instead uses an event loop to execute multiple tasks concurrently.

```py
import asyncio 

async def async_io_task(name, duration): 
	print(f"Task {name} starting") 
	await asyncio.sleep(duration) 
	print(f"Task {name} completed") 

async def main(): 
	task1 = asyncio.create_task(async_io_task("A", 2)) 
	task2 = asyncio.create_task(async_io_task("B", 4)) 
	await task1 
	await task2 
	asyncio.run(main())
```

Await is like non-blocking sleep function. This can be used when performing tasks like getting prices from different exchanges or scraping websites where they may be delays in getting a response. Another task can start executing whilst one is waiting for a response.

### Regex

The re module provides functions for using regular expressions:

```py
import re 

text = "The price of apples is $5." 
match = re.search(r'\$\d+', text) 
if match: 
	print("Found:", match.group()) # Output: Found: $5 
else: 
	print("Not found.")

```

Here group() returns the whole match group. Subgroups defined using () can also be retuned with group(n)

```py
# Finding All Matches: 
re.findall()

# Splitting a String: 
re.split()

# Substituting Text: 
re.sub()

# Compiling Regular Expressions: 
re.compile()


```


## Data Science

## Pandas


Pandas uses DataFrames as its main data type. These are:

* tables (2D)
* Have labels for row/columns
* Can have different types
* Can grow/shrink

```py
import pandas as pd

data = [["Timmah",21],["Peter",12],["Byron",72]]
df = pd.DataFrame(data,columns=["name","age"])

>>> df # rendered natievly in Jupyter Notebooks
```

 ![](Notes/image.png)<!-- {"width":149} -->

### DataFrame meta / stats

```py
df.info()

<class 'pandas.core.frame.DataFrame'>
RangeIndex: 3 entries, 0 to 2
Data columns (total 3 columns):
 #   Column  Non-Null Count  Dtype 
---  ------  --------------  ----- 
 0   Name    3 non-null      object
 1   Age     3 non-null      int64 
 2   City    3 non-null      object
dtypes: int64(1), object(2)
memory usage: 204.0+ bytes
```


```py
# We get some stats on the dats for free
print(df.describe())

        Age
count   3.0
mean   30.0
std     5.0
min    25.0
25%    27.5
50%    30.0
75%    32.5
max    35.0

```

### Retrieve a single column

```py
# Get a single column
column = df["Name"]
print (column)

0      Alice
1        Bob
2    Charlie
Name: Name, dtype: object

# Columns are type Series
print(type(column))

pandas.core.series.Series

# Read a CSV into a DataFrame
df = pd.read_csv("data.csv")

```


### Process a CSV 

Can process historic data from Yahoo finance by getting url from download link

```py
df = pd.read_csv("https://query1.finance.yahoo.com/v7/finance/download/GOOG?period1=1092873600&period2=1716940800&interval=1d&events=history&includeAdjustedClose=true")

df
```

![](Notes/image%202.png)<!-- {"width":493} -->

![](Notes/image%203.png)<!-- {"width":493} -->

note: Standard deviation indicates how much individual data points are dispersed from the mean, It’s a way of measuring volatility and hence also risk (/potential return). A portfolio with a load STD is more consistent.

### Adding columns
Use the Pandas pct change to create a Daily Return column / series from the adjusted close

```py
# Calculate the daily return
df['Daily Return'] = df['Adj Close'].pct_change()
```

```py
# Create a 50-day moving average column
df['50day-ma'] = df['Daily Return'].rolling(window=50).mean()
```


### Missing Data
Financial time series often have missing data due to non-trading days or other reasons. Handling missing data in pandas involves identifying the missing values and then deciding on the appropriate method to handle them:

	* 	**Identifying**: isnull(), notnull()
	* 	**Dropping**: dropna()
	* 	**Filling**: fillna(), ffill(), bfill()
	* 	**Interpolating**: interpolate()
	* 	**Replacing**: replace()

```py
# Forward fill to handle non-trading days
df_filled = df.ffill()
print(df_filled)

# Interpolating missing values
df_interpolated = df.interpolate(method='linear')
print(df_interpolated)
```

Forward-fill may be appropriate to fill in when there have been non-trading days and is all most likely to be a reasonable estimate. Its also computationally fast

Interpolation can be useful for smoothing transitions between known data-points. Can be more accurate where data is changing gradually.

## NumPy

NumPy is most commonly used for working with arrays of data. NumPy, primarily focused on numerical computing, is renowned for its support of large, multi-dimensional arrays and matrices. Its commonly used for:

* Analyzing financial data
* Modeling financial instruments
* Optimizing portfolios

Unlike DataFrames, numpy Arrays (ndarray) are homogenous (single type only)

```py
import numpy as np

# Create numpy array
numbers = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])

# Perform some in-buit operations
mean = np.mean(numbers)
sines = np.sin(numbers)
```

```py
# Import csv to array
prices = np.genfromtxt("stock_prices.csv", delimiter=",")

# Calculate returns
returns = np.diff(prices)
```


## Matplotlib
Plotting / data vis library for Python.

```py
import numpy as np
import matplotlib.pyplot as plt

# x: 100 datapoints from 0->2pi
x = np.linspace(0,2*np.pi,100)
# y: sin(x)
y = np.sin(x)

# Plot and show chart
plt.plt(x,y)
ply.show()
```

Plot a random scatter graph:

```py
import random

x = [random.random() for _ in range(10)]
y = [random.random() for _ in range(10)]

plt.scatter(x,y)
```

#### Quick list comprehension refresher

```py
# Creates an array 0...9, iterates over it to 
# make a new list multiplying each element by 10
y = [r*10 for r in range(10)]
```

### Histogram plot

```py

# get historic price data from URL
df = pd.read_csv("https://query1.finance.yahoo.com/v7/finance/download/GOOG?period1=1092873600&period2=1716940800&interval=1d&events=history&includeAdjustedClose=true", index_col="Date", parse_dates=True)

# Add Daily Return Column
df['Returns'] = df['Adj Close'].pct_change()

# Drop rows with empty values
df = df.dropna()

# Render frequency histogram
plt.hist(df['Returns'], bins=50, alpha=0.6, color='blue')

# Add titles and labels
plt.title("Histogram of Returns")
plt.xlabel("Return")
plt.ylabel("Frequency")
```

![](Notes/image%204.png)


### Bollinger bands

These are to measure market volatility and to identify potential overbought or oversold conditions. There are three lines :

* Middle Band (20 day moving average or Simple Moving Avg / SMA) 
* SMA + (2 x STD of Price)
* SMA - (2 x STD of Price)

```py

# Create columns for the SMA, Upper and Lower bands
df['SMA'] = df['Adj Close'].rolling(window=20).mean()
df['STD'] = df['Close'].rolling(window=20).std()
df['Upper Band'] = df['SMA'] + (df['STD'] * 2)
df['Lower Band'] = df['SMA'] - (df['STD'] * 2)

plt.figure(figsize=(12,6))
plt.plot(df['Adj Close'], label='BOOF Adj Close', color='blue')
plt.plot(df['SMA'], label='Middle Band', color='red')
plt.plot(df['Upper Band'], label='Upper Band', color='green')
plt.plot(df['Lower Band'], label='Lower Band', color='green')
plt.fill_between(df, df['Upper Band'], df['Lower Band'], color='grey', alpha=0.3)
plt.title('BOOF Bollinger Bands')
plt.xlabel('Date')
plt.ylabel('Price')
plt.legend()
plt.show()
```

