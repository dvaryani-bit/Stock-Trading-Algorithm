import matplotlib.pyplot as plt
import yfinance as yf
import datetime
from sklearn.linear_model import LinearRegression
import pandas


stock_obj = yf.Ticker("AAPL")

#print(stock_obj.info)

#average of a day's high and low
data = stock_obj.history(period="max")[['High', 'Low']]
data['Price'] = (data['High'] + data['Low']) / 2
del data['High']
del data['Low']
data.reset_index(inplace=True)

#timespan of data selected
now = datetime.datetime.now()
six_months_before = now - datetime.timedelta(days=183)
twelve_months_before = now - datetime.timedelta(days=365)
eighteen_months_before = now - datetime.timedelta(days=548)
one_week_before = now - datetime.timedelta(days=7)

SELECTED_TIMESPAN = six_months_before

data = data[data['Date'] > SELECTED_TIMESPAN]
average = sum(data['Price'])/len(data['Price'])
pop_standard_deviation = (sum([(x-average)**2 for x in data['Price']])/len(data['Price']))**0.5
data['Deviation'] = data['Price'].apply(lambda x: 100*(x-average)/average)
#data['Variation Coefficient'] = data['Deviation'].apply(lambda x: round(x/pop_standard_deviation, 1))

MOVING_AVERAGE_SPAN = 30

data['Moving Average'] = data['Price'].rolling(MOVING_AVERAGE_SPAN, center=True).mean()
data['Moving Standard Deviation'] = data['Price'].rolling(MOVING_AVERAGE_SPAN, center=True).std()

# first and last Moving Avg nulls in edges due to MOVING_AVERAGE_SPAN range
# getting more moving average data in edges with shorter span
EDGES_MOVING_AVERAGE_SPAN = 7

data_null = data[data['Moving Average'].isnull()]
null_indexes = data_null.index.tolist()
for i in range(len(null_indexes)):
    if null_indexes[i+1] != null_indexes[i] + 1:
        breakpoint = i
        break
data_null_1 = pandas.concat([data_null.loc[null_indexes[:breakpoint+1]], data.loc[[null_indexes[breakpoint] + i for i in range(1, EDGES_MOVING_AVERAGE_SPAN)]]])
data_null_2 = pandas.concat([data.loc[[null_indexes[breakpoint+1] - i for i in range(EDGES_MOVING_AVERAGE_SPAN, 0, -1)]], data_null.loc[null_indexes[breakpoint+1:]]])

data_null_1['Moving Average'] = data_null_1['Price'].rolling(EDGES_MOVING_AVERAGE_SPAN, center=True).mean()
data_null_1['Moving Standard Deviation'] = data_null_1['Price'].rolling(EDGES_MOVING_AVERAGE_SPAN, center=True).std()
data_null_2['Moving Average'] = data_null_2['Price'].rolling(EDGES_MOVING_AVERAGE_SPAN, center=True).mean()
data_null_2['Moving Standard Deviation'] = data_null_2['Price'].rolling(EDGES_MOVING_AVERAGE_SPAN, center=True).std()

data_null = pandas.concat([data_null_1, data_null_2]).loc[null_indexes]

for i in null_indexes:
    data.loc[i, 'Moving Average'] = data_null.loc[i, 'Moving Average']
    data.loc[i, 'Moving Standard Deviation'] = data_null.loc[i, 'Moving Standard Deviation']

# heuristics to eliminate noise
data['Moving Deviation'] = 100*(data['Price'] - data['Moving Average']) / data['Moving Average']
data['Moving Variation Coefficient'] = round(data['Moving Deviation']/data['Moving Standard Deviation'], 1)

data.to_csv("output.csv")

var_coeffs = data['Moving Variation Coefficient'].tolist()
#var_coeffs = data['Variation Coefficient'].tolist()
var_coeffs.sort()
print(var_coeffs)
#VAR_SELECTED = 1.1


data2 = data[data['Moving Variation Coefficient'] >= 0.8]
data3 = data[data['Moving Variation Coefficient'] <= -0.7]


### resistance
model_r = LinearRegression()
x = []
c = 0
for i in range(len(data2)):
    x.append([c])
    c += 1
model_r.fit(x, data2['Price'])
r_sq_r = model_r.score(x, data2['Price'])
print(f"coefficient of determination: {r_sq_r}")
print(f"intercept: {model_r.intercept_}")
print(f"slope: {model_r.coef_}")
data2['Fit'] = ""
c = 0
for i in data2.index.values:
    #print(i)
    data2.loc[i, 'Fit'] = c*model_r.coef_ + model_r.intercept_
    c += 1

#x_resistance = [data.iloc[0]['Date'], data2.iloc[-1]['Date']]
#print(x_resistance)
#y_resistance = [model_r.intercept_, data2.iloc[-1]['Price']]

#### support

model_s = LinearRegression()
x = []
c = 0
for i in range(len(data3)):
    x.append([c])
    c += 1
model_s.fit(x, data3['Price'])
r_sq_s = model_s.score(x, data3['Price'])
print(f"coefficient of determination: {r_sq_s}")
print(f"intercept: {model_s.intercept_}")
print(f"slope: {model_s.coef_}")
data3['Fit'] = ""
c = 0
for i in data3.index.values:
    # print(i)
    data3.loc[i, 'Fit'] = c * model_s.coef_ + model_s.intercept_
    c += 1



plt.figure(figsize=(50, 25), dpi=100)
plt.plot(data['Date'], data['Price'], linewidth=5, label = '1')
#plt.plot(x_resistance, y_resistance, label = '2', linewidth = 10)
plt.plot(data2['Date'], data2['Fit'], label = '2', linewidth = 10)
plt.plot(data3['Date'], data3['Fit'], label = '2', linewidth = 10)
plt.xlabel("Date", fontsize=40)  # mention the timespan
plt.ylabel("Stock Price", fontsize=40)
plt.title("{} Time Series".format(stock_obj.ticker), fontsize=40)  # mention the stock ticker
plt.tick_params(axis='x', labelsize=40)
plt.tick_params(axis='y', labelsize=40)
#plt.text(data['Date'],data['Price'], data.index())
#for x,y in zip(data['Date'], data['Price']):
#    plt.annotate(data['Price'], (x, y), textcoords="offset points", xytext=(0,10),ha='center')
plt.legend()
plt.show()



