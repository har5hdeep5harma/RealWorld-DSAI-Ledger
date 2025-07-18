
If you’ve done any data analysis in Python, you’ve probably run across [Pandas](http://pandas.pydata.org/pandas-docs/stable/), a fantastic analytics library written by Wes McKinney. By conferring dataframe analysis functionality to Python, Pandas has effectively put Python on the same footing as some of the more established analysis tools, such as R or SAS.

Unfortunately, early on, Pandas had gotten a nasty reputation for being “slow”. It’s true that your Pandas code is unlikely to reach the calculation speeds of, say, fully optimized raw C code. However, the good news is that for most applications, well-written Pandas code is _fast enough_; and what Pandas lacks in speed, it makes up for in being powerful and user-friendly.

In this post, we’ll review the efficiency of several methodologies for applying a function to a Pandas DataFrame, from slowest to fastest:  
1. Crude looping over DataFrame rows using indices  
2. Looping with `iterrows()`  
3. Looping with `apply()`  
4. Vectorization with Pandas series  
5. Vectorization with NumPy arrays

For our example function, we’ll use the [Haversine](https://en.wikipedia.org/wiki/Haversine_formula) (or Great Circle) distance formula. Our function takes the latitude and longitude of two points, adjusts for Earth’s curvature, and calculates the straight-line distance between them. The function looks something like this:

```
import numpy as np

# Define a basic Haversine distance formula
def haversine(lat1, lon1, lat2, lon2):
    MILES = 3959
    lat1, lon1, lat2, lon2 = map(np.deg2rad, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1 
    dlon = lon2 - lon1 
    a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
    c = 2 * np.arcsin(np.sqrt(a)) 
    total_miles = MILES * c
    return total_miles
```

To test our function on real data, we’ll use a dataset containing the coordinates of all hotels in New York state, sourced from [Expedia’s developer site](http://developer.ean.com/). We’ll calculate the distance between each hotel and a sample set of coordinates (which happen to belong to a fantastic little shop called the [Brooklyn Superhero Supply Store](https://www.superherosupplies.com/) in NYC).

You can download the dataset, and the Jupyter notebook containing the functions used in this blog, [here](https://github.com/sversh/pycon2017-optimizing-pandas).

This post is based on my PyCon talk, which you can watch [here](https://www.youtube.com/watch?v=HN5d490_KKk).

# Crude looping in Pandas, or That Thing You Should Never Ever Do

To start, let’s quickly review the fundamentals of Pandas data structures. The basic Pandas structures come in two flavors: a **DataFrame** and a **Series**. A DataFrame is a two-dimensional **array** with labeled axes. In other words, a DataFrame is a matrix of rows and columns that have labels — column names for columns, and index labels for rows. A single column or row in a Pandas DataFrame is a Pandas series — a one-dimensional array with axis labels.

Just about every Pandas beginner I’ve ever worked with (including yours truly) has, at some point, attempted to apply a custom function by looping over DataFrame rows one at a time. The advantage of this approach is that it is consistent with the way one would interact with other iterable Python objects; for example, the way one might loop through a list or a tuple. Conversely, the downside is that a crude loop, in Pandas, is the slowest way to get anything done. Unlike the approaches we will discuss below, crude looping in Pandas does not take advantage of any built-in optimizations, making it extremely inefficient (and often much less readable) by comparison.

For example, one might write something like this:

```
# Define a function to manually loop over all rows and return a series of distances
def haversine_looping(df):
    distance_list = []
    for i in range(0, len(df)):
        d = haversine(40.671, -73.985, df.iloc[i]['latitude'], df.iloc[i]['longitude'])
        distance_list.append(d)
    return distance_list
```

To get a sense of the time required to execute the function above, we’ll use the `%timeit` command. `%timeit` is a “[magic](http://ipython.readthedocs.io/en/stable/interactive/magics.html)” command built specifically for use with [Jupyter notebooks](http://jupyter.org/). (All magic commands start with a `%`sign if the command is to apply to a single line, and a `%%` to apply to an entire Jupyter cell). `%timeit` will run a function multiple times, and print the average and standard deviation of the runtimes it obtained. Of course, the runtime obtained by `%timeit` will not necessarily be the same on every system running the function. Still, it can offer a useful benchmarking tool for comparing the runtime of different functions on the same system and dataset.

```
%%timeit

# Run the haversine looping function
df['distance'] = haversine_looping(df)
```

This returns the following result:

>645 ms ± 31 ms per loop (mean ± std. dev. of 7 runs, 1 loop each)

Our crude looping function took about 645 ms to run, with a standard deviation of 31 ms. This may seem fast, but it’s actually quite slow, considering the function only needed to process some 1,600 rows. Let’s look at how we can improve this unfortunate state of affairs.
# Looping with iterrows()

A better way to loop through rows, if loop you must, is with the [iterrows()](https://pandas.pydata.org/pandas-docs/stable/generated/pandas.DataFrame.iterrows.html)`method. `[iterrows()](https://pandas.pydata.org/pandas-docs/stable/generated/pandas.DataFrame.iterrows.html) is a generator that iterates over the rows of the dataframe and returns the index of each row, in addition to an object containing the row itself. `iterrows()` is optimized to work with Pandas dataframes, and, although it’s the least efficient way to run most standard functions (more on that later), it’s a significant improvement over crude looping. In our case, [iterrows()](https://pandas.pydata.org/pandas-docs/stable/generated/pandas.DataFrame.iterrows.html) solves the same problem almost four times faster than manually looping over rows.

```
%%timeit

# Haversine applied on rows via iteration
haversine_series = []
for index, row in df.iterrows():
    haversine_series.append(haversine(40.671, -73.985, row['latitude'], row['longitude']))
df['distance'] = haversine_series
```

>166 ms ± 2.42 ms per loop (mean ± std. dev. of 7 runs, 1 loop each)

# Better looping using the apply method

An even better option than `iterrows()` is to use the `[apply()](https://pandas.pydata.org/pandas-docs/stable/generated/pandas.DataFrame.apply.html)` method, which applies a function along a specific axis (meaning, either rows or columns) of a DataFrame. Although `apply()` also inherently loops through rows, it does so much more efficiently than `iterrows()` by taking advantage of a number of internal optimizations, such as using iterators in Cython.

We use an anonymous lambda function to `apply` our Haversine function on each row, which allows us to point to specific cells within each row as inputs to the function. The lambda function includes the `axis` parameter at the end, in order to specify whether Pandas should apply the function to rows (`axis = 1`) or columns (`axis = 0`).

```
%%timeit

# Timing apply on the Haversine function
df['distance'] = df.apply(lambda row: haversine(40.671, -73.985, row['latitude'], row['longitude']), axis=1)
```

>90.6 ms ± 7.55 ms per loop (mean ± std. dev. of 7 runs, 10 loops each)

Swapping `apply()` for `iterrows()` has roughly halved the runtime of the function!

To get more insight into what’s actually taking up runtime within our function, we can run a [line profiler tool](https://github.com/rkern/line_profiler) (the `%lprun` magic command in Jupyter).

```
# Haversine applied on rows with line profiler
%lprun -f haversine df.apply(lambda row: haversine(40.671, -73.985, row['latitude'], row['longitude']), axis=1)
```

This returns the following:
<img src="https://miro.medium.com/v2/resize:fit:1400/format:webp/1*q0cYC_LTZDiPZ-LpJCQ9Hg.png">

We can glean a number of useful insights from this information. For example, the parts of the function that are doing the trigonometric calculations account for nearly half of the total runtime. Therefore, if we were looking to optimize individual components of the function, we might want to start there. For now, it is particularly significant to note that each line is being hit a whopping 1,631 times — a result of `apply()` iterating over each row. If we could cut down on the amount of repetitive work being done, we could cut down on the overall runtime. That’s where vectorization offers a vastly more efficient alternative.
# Vectorization over Pandas series

To understand how we can reduce the amount of iteration performed by the function, recall that the fundamental units of Pandas, DataFrames and series, are both based on arrays. The inherent structure of the fundamental units translates to built-in Pandas functions being designed to operate on entire arrays, instead of sequentially on individual values (referred to as **scalars**). **Vectorization** is the process of executing operations on entire arrays.

Pandas includes a generous collection of vectorized functions for everything from mathematical operations to aggregations and string functions (for an extensive list of available functions, check out the [Pandas docs](https://pandas.pydata.org/pandas-docs/stable/index.html)). The built-in functions are optimized to operate specifically on Pandas series and DataFrames. As a result, using vectorized Pandas functions is almost always preferable to accomplishing similar ends with custom-written looping.

So far, we’ve only been passing scalars to our Haversine function. All of the functions being used within the Haversine function, however, are also able to operate on arrays. This makes the process of vectorizing our distance function quite simple: instead of passing individual scalar values for latitude and longitude to it, we’re going to pass it the entire series (columns). This will allow Pandas to benefit from the full set of optimizations available for vectorized functions, including, notably, performing all the calculations on the entire array simultaneously.

```
%%timeit 

# Vectorized implementation of Haversine applied on Pandas series
df['distance'] = haversine(40.671, -73.985, df['latitude'], df['longitude'])
```

>1.62 ms ± 41.5 µs per loop (mean ± std. dev. of 7 runs, 1000 loops each)

We’ve achieved more than a 50-fold improvement over the `apply()` method, and more than a 100-fold improvement over `iterrows()` by vectorizing the function — and we didn’t need to do anything but change the input type!

Let’s peek under the hood and see what the function is doing:
<img src="https://miro.medium.com/v2/resize:fit:1400/format:webp/1*8jWZlViD3ocHETMriihxvQ.png">

Notice that whereas `apply()` was administering the function 1,631 times, the vectorized version only applies the function exactly once, since it is being applied to the entire array simultaneously. That’s where the serious time savings are coming from.
# Vectorization with NumPy arrays

At this point, we could choose to call it a day; vectorizing over Pandas series achieves the overwhelming majority of optimization needs for everyday calculations. However, if speed is of highest priority, we can call in reinforcements in the form of the NumPy Python library.

The [NumPy library](http://www.numpy.org/), which describes itself as a “fundamental package for scientific computing in Python”, performs operations under the hood in optimized, pre-compiled C code. Like Pandas, NumPy operates on array objects (referred to as ndarrays); however, it leaves out a lot of overhead incurred by operations on Pandas series, such as indexing, data type checking, etc. As a result, operations on NumPy arrays can be significantly faster than operations on Pandas series.

NumPy arrays can be used in place of Pandas series when the additional functionality offered by Pandas series isn’t critical. For example, the vectorized implementation of our Haversine function doesn’t actually use indexes on the latitude or longitude series, and so not having those indexes available will not cause the function to break. By comparison, had we been doing operations like DataFrame joins, which require referring to values by index, we might want to stick to using Pandas objects.

We convert our latitude and longitude arrays from Pandas series to NumPy arrays simply by using the `values` method of the series. As with vectorization on the series, passing the NumPy array directly into the function will lead Pandas to apply the function to the entire vector.

```
%%timeit

# Vectorized implementation of Haversine applied on NumPy arrays
df['distance'] = haversine(40.671, -73.985, df['latitude'].values, df['longitude'].values)
```

>370 µs ± 18 µs per loop (mean ± std. dev. of 7 runs, 1000 loops each)

Running the operation on NumPy array has achieved another four-fold improvement. All in all, we’ve refined the runtime from over half a second, via looping, to a third of a millisecond, via vectorization with NumPy!
# Summary

The scoreboard below summarizes the results. Although vectorization with NumPy arrays resulted in the fastest runtime, it was a fairly marginal improvement over the effect of vectorization with Pandas series, which resulted in a whopping 56x improvement over the fastest version of looping.
<img src="https://miro.medium.com/v2/resize:fit:1400/format:webp/1*p4zjrqG97C4bFmOXU5UQog.png">

This brings us to a few basic conclusions on optimizing Pandas code:  
1. Avoid loops; they’re slow and, in most common use cases, unnecessary.  
2. If you must loop, use apply(), not iteration functions.  
3. Vectorization is usually better than scalar operations. Most common operations in Pandas can be vectorized.  
4. Vector operations on NumPy arrays are more efficient than on native Pandas series.

The above does not, of course, make up a comprehensive list of all possible optimizations for Pandas. More adventurous users might consider, for example, further rewriting the function in [Cython](http://cython.readthedocs.io/en/latest/src/quickstart/overview.html), or attempting to optimize the individual components of the function. However, these topics are beyond our scope.

Crucially, before embarking on a grand optimization adventure, make sure that the functions you are optimizing are actually the functions that you will want to use in the long run. To quote the immortal words of Donald Knuth: “[Premature optimization is the root of all evil.](https://xkcd.com/1691/)”

