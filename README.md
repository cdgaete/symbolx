# Symbolx

Symbolx is a Python library for collecting, organizing, and analyzing multi-dimensional data from optimization model outputs. It provides a flexible and powerful toolkit for data scientists and researchers working with complex data structures, particularly in fields like energy systems modeling.

## Installation

You can install Symbolx using pip:

```bash
pip install symbolx
```

## Quick Start

Here are some examples to get you started with Symbolx:

### 1. Collecting Data from Optimization Model Outputs

```python
import symbolx as syx
from symbolx import DataCollection

# Initialize DataCollection
DC = DataCollection()

# Add a collector for Feather files
DC.add_collector(collector_name='opt_model', parser=syx.symbol_parser_feather, loader=syx.load_feather)
DC.add_folder(collector_name='opt_model', './raw_model_output')
DC.add_custom_attr(collector_name='opt_model', with_='pandas')

# Acquire data
DC.adquire(id_integer=True, zip_extension=None)
```

### 2. Creating Symbols from Collected Data

```python
from symbolx import SymbolsHandler, Symbol

# Initialize SymbolsHandler
SH = SymbolsHandler(method='object', obj=DC)

# Create symbols
var1 = Symbol(name='VAR1', symbol_handler=SH)
var2 = Symbol(name='VAR2', value_type='v', symbol_handler=SH)
var3 = Symbol(name='VAR3', value_type='m', symbol_handler=SH)
```

### 3. Basic Symbol Operations

```python
# Arithmetic operations
result = var1 + var2
result = var1 * 2

# Comparison operations
mask = var1 > 10

# Reduction operations
sum_var1 = var1.dimreduc(dim='h', aggfunc='sum')
```

### 4. Querying and Filtering Data

```python
# Query based on metadata attributes
filtered_var1 = var1.query("country == 'Germany' and year > 2020")

# Shrink symbol based on specific coordinates
shrunk_var1 = var1.shrink(tech=['wind', 'solar'], year=[2025, 2030])
```

### 5. Working with Metadata

```python
# Get metadata for a specific scenario
scenario_info = var1.id_info(1)

# Get summary of metadata across all scenarios
metadata_summary = var1.summary
```

### 6. Exporting Data

```python
# Export to Feather file
var1.to_feather('var1_data.feather')

# Convert to pandas DataFrame
df = var1.to_pandas()

# Convert to polars DataFrame
pl_df = var1.to_polars()
```

### 7. Advanced Operations

```python
# Transform data based on metadata attributes
transformed_var1 = var1.transform(metadata_attr=['country', 'year'], function='sum')

# Calculate difference from a reference scenario
diff_var1 = var1.refdiff(reference_id=1)

# Group data by metadata attributes
grouped_var1 = var1.groupby(['country', 'year'])
```
