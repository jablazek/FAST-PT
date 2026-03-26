import numpy as np
import sys
import sys

class CacheManager:
    """Unified cache manager for FASTPT with memory efficiency features"""
    
    def __init__(self, max_size_mb=500, dump_cache=True):
        """Initialize cache with optional maximum size in MB"""
        self.cache = {}
        self.hit_counts = {}  # Track hits per cache item
        self.cache_size = 0
        self.max_size_bytes = max_size_mb * 1024 * 1024
        #^^ 1000 MB = 1000*1024 KB = 1000*1024*1024 bytes (1024 instead of 1000 due to binary memory 2^10=1024)
        self.hits = 0
        self.misses = 0
        self.dump_cache = dump_cache
        self.current_P_hash = 0

    def measure_actual_size(self):
        """Measure actual memory usage of the cache"""
        # First try pympler if available
        try:
            from pympler import asizeof
            actual_size = asizeof.asizeof(self.cache) / (1024 * 1024)
            return f"{actual_size:.2f}"
        except ImportError:
            pass
        
        # Fallback to sys.getsizeof (less accurate for complex objects)
        import sys
        basic_size = sys.getsizeof(self.cache) / (1024 * 1024)
        return f"Basic cache size (sys.getsizeof): {basic_size:.2f} MB"
    
    def _get_array_size(self, arr):
        """Calculate size of objects in bytes, accounting for Python objects"""
        if isinstance(arr, np.ndarray):
            return arr.nbytes
        elif isinstance(arr, (tuple, list)):
            # Account for container overhead (~56 bytes per tuple/list in CPython)
            container_overhead = sys.getsizeof(arr) - sum(sys.getsizeof(0) for _ in range(len(arr)))
            return container_overhead + sum(self._get_array_size(item) for item in arr)
        elif isinstance(arr, (int, float, str, bool)):
            return sys.getsizeof(arr)
        elif arr is None:
            return sys.getsizeof(None)
        else:
            try:
                return sys.getsizeof(arr)
            except:
                return 64  # Default estimate if sys.getsizeof fails
        
    def get(self, category, hash_key):
        """Get an item from cache using category and arguments as key"""
        key = (category, hash_key)
        if key in self.cache:
            self.hits += 1
            # Track hit counts per item
            if key in self.hit_counts:
                self.hit_counts[key] += 1
            else:
                self.hit_counts[key] = 1
            return self.cache[key]
        self.misses += 1
        return None
    
    def set(self, value, category, hash_key, P_hash):
        """Store an item in cache using category and arguments as key"""
        key = (category, hash_key)
        key_size = self._get_array_size(key)

        old_size = 0
        if key in self.cache:
            old_val = self.cache[key]
            old_size = self._get_array_size(old_val)
        else:
            self.hit_counts[key] = 0
        
        value_size = self._get_array_size(value)
        total_size = key_size + value_size
        
        if self.dump_cache and P_hash is not None and P_hash != self.current_P_hash:
            self.cache.clear()
            self.hit_counts.clear()
            self.cache_size = 0
            self.current_P_hash = P_hash
            self.cache_size = sum(self._get_array_size(k) + self._get_array_size(v) 
                                for k, v in self.cache.items())
        
        elif self.max_size_bytes > 0 and (self.cache_size - old_size + total_size) > self.max_size_bytes:
            import warnings
            warnings.warn(f"Cache size exceeds maximum limit of {self.max_size_bytes / (1024 * 1024):.2f} MB. Evicting items.")
            self._evict(total_size - old_size)
        
        self.cache[key] = value
        self.cache_size = self.cache_size - old_size + total_size
        return value
    
    def _evict(self, required_size):
        """Evict items from cache until there's room for required_size"""
        items = list(self.cache.items())
        np.random.shuffle(items)
    
        freed = 0
        for key, value in items:
            if freed >= required_size:
                break
            
            key_size = self._get_array_size(key)
            value_size = self._get_array_size(value)
            total_size = key_size + value_size
        
            del self.cache[key]
            if key in self.hit_counts:
                del self.hit_counts[key]
            
            self.cache_size -= total_size
            freed += total_size
    
    def clear(self):
        """Clear the entire cache"""
        self.cache.clear()
        self.hit_counts.clear()
        self.cache_size = 0
    
    def stats(self):
        """Return statistics about the cache usage"""
        # Calculate key and value sizes separately
        key_size = 0
        value_size = 0
        
        for key, value in self.cache.items():
            key_size += self._get_array_size(key)
            value_size += self._get_array_size(value)
        
        # Total size
        total_size_bytes = self.cache_size
        total_size_mb = total_size_bytes / (1024 * 1024)
        key_size_mb = key_size / (1024 * 1024)
        value_size_mb = value_size / (1024 * 1024)
        
        # Calculate hit rate
        total_accesses = self.hits + self.misses
        hit_rate = self.hits / total_accesses if total_accesses > 0 else 0
        
        # Max cache size in MB
        max_size_mb = self.max_size_bytes / (1024 * 1024) if self.max_size_bytes > 0 else float('inf')
        
        return {
            'items': len(self.cache),
            'size_bytes': total_size_bytes,
            'Pympler_size': self.measure_actual_size(),
            'size_mb': total_size_mb,
            'key_size_mb': key_size_mb,
            'value_size_mb': value_size_mb,
            'key_percent': (key_size / total_size_bytes) * 100 if total_size_bytes > 0 else 0,
            'value_percent': (value_size / total_size_bytes) * 100 if total_size_bytes > 0 else 0,
            'max_size_mb': max_size_mb,
            'percent_full': (total_size_bytes / self.max_size_bytes) * 100 if self.max_size_bytes > 0 else 0,
            'hits': self.hits,
            'misses': self.misses,
            'hit_rate': hit_rate
        }

    def __repr__(self):
        """Return a string representation of the cache"""
        stats = self.stats()
        result = [
            f"CacheManager: {stats['size_mb']:.2f}MB/{stats['max_size_mb']:.2f}MB used",
            f"Memory breakdown: Keys: {stats['key_size_mb']:.2f}MB ({stats['key_percent']:.1f}%), Values: {stats['value_size_mb']:.2f}MB ({stats['value_percent']:.1f}%)",
            f"Items: {stats['items']}, Hit rate: {stats['hit_rate']:.2%}"
        ]
    
        if stats['items'] > 0:
            # Group items by category
            categories = {}
            for key, value in self.cache.items():
                category, hash_key = key
                if category not in categories:
                    categories[category] = []
                hit_count = self.hit_counts.get(key, 0)
                categories[category].append((hash_key, value, hit_count))
        
            result.append("\nCached items by category:")
            for category, items in sorted(categories.items()):
                result.append(f"\n  Category: {category} ({len(items)} items)")
            
                for hash_key, value, hit_count in items:
                    # Format the value representation
                    if isinstance(value, np.ndarray):
                        shape_str = f"shape={value.shape}"
                        dtype_str = f"dtype={value.dtype}"
                        if value.size > 5:
                            # For large arrays, show a few elements and shape/dtype
                            val_repr = f"ndarray({shape_str}, {dtype_str}, first few: {value.flat[:3]}...)"
                        else:
                            # For small arrays, show all values
                            val_repr = f"ndarray({shape_str}, {dtype_str}, values: {value})"
                    elif isinstance(value, (list, tuple)):
                        val_type = type(value).__name__
                        length = len(value)
                        if length > 3:
                            first_items = []
                            for i, item in enumerate(value[:2]):
                                if isinstance(item, np.ndarray):
                                    first_items.append(f"ndarray(shape={item.shape})")
                                else:
                                    first_items.append(str(item))
                            val_repr = f"{val_type} of {length} items: [{', '.join(first_items)}...]"
                        else:
                            val_repr = f"{val_type} of {length} items"
                    else:
                        val_repr = str(value)
                
                    # Truncate representation if too long
                    if len(val_repr) > 100:
                        val_repr = val_repr[:97] + "..."
                
                    # Format hash key based on its magnitude
                    if isinstance(hash_key, int):
                        if abs(hash_key) > 1000000000:  # Large hash values
                            hash_repr = f"hash:{hash_key:.3e}"  # Scientific notation
                        else:
                            hash_repr = f"hash:{hash_key}"
                    else:
                        hash_repr = str(hash_key)
                
                    result.append(f"    • {hash_repr} [hits: {hit_count}]: {val_repr}")
    
        return "\n".join(result)