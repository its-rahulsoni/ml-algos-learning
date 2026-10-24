# Python Method Types vs Java

Python has three categories of methods. The distinction maps onto Java concepts, but not perfectly.

## 1. Instance methods (the default, and what you've seen so far)

An **instance method** is a method that operates on a specific object (instance) of the class — it needs `self` because it reads or modifies *that particular object's* data.

```python
class TitanicCleaner:
    def __init__(self, csv_path):
        self.csv_path = csv_path

    def load(self):              # <- instance method
        self.df = pd.read_csv(self.csv_path)   # uses THIS object's csv_path
        return self.df
```

Every method you've seen in `TitanicCleaner` so far (`load`, `handle_missing_age`, `clean`, etc.) is an instance method. You call it on an object:

```python
cleaner = TitanicCleaner("titanic.csv")
cleaner.load()   # operates on cleaner's own self.csv_path
```

**Java equivalent:** an ordinary (non-static) method — `public void load() { ... }`. The implicit `this` in that Java method is exactly what Python's explicit `self` is doing.

## 2. Class methods — `@classmethod`

A **class method** doesn't operate on one specific object — it operates on the *class itself*. Instead of `self`, its first parameter is `cls` (the class, not an instance). You mark it with a `@classmethod` decorator:

```python
class TitanicCleaner:
    default_path = "titanic.csv"

    def __init__(self, csv_path):
        self.csv_path = csv_path

    @classmethod
    def from_default(cls):
        # cls here IS the TitanicCleaner class itself
        return cls(cls.default_path)   # equivalent to TitanicCleaner("titanic.csv")
```

You'd call this on the class, not an instance:

```python
cleaner = TitanicCleaner.from_default()
```

This is commonly used as an alternative constructor pattern — "build me an instance a different way" — since Python only allows one `__init__`.

**Java equivalent:** there isn't a clean one-to-one match. The closest thing is a `static` factory method that returns `new TitanicCleaner(...)`, but Java's `static` methods don't receive an implicit reference to the class the way `cls` does in Python. `cls` is genuinely a parameter Python passes for you, just like `self` is for instance methods — Java has no equivalent mechanic, it's just convention there.

## 3. Static methods — `@staticmethod`

A **static method** takes neither `self` nor `cls`. It's just a regular function that happens to live inside the class namespace because it's topically related, but it doesn't touch the object or the class at all:

```python
class TitanicCleaner:
    @staticmethod
    def extract_title(name_string):
        # doesn't need self or cls — pure function, just organized under this class
        import re
        match = re.search(r",\s*([^\.]+)\.", name_string)
        return match.group(1) if match else None
```

Called via the class or an instance, same result either way:

```python
TitanicCleaner.extract_title("Braund, Mr. Owen Harris")
cleaner.extract_title("Braund, Mr. Owen Harris")   # also works
```

**Java equivalent:** this one *does* match cleanly — it's exactly Java's `static` method. No implicit `this`, no access to instance data, just a function namespaced under the class.

## Summary table

| | Python | First parameter | Called via | Java equivalent |
|---|---|---|---|---|
| Instance method | plain `def method(self):` | `self` (the object) | `instance.method()` | ordinary (non-static) method, implicit `this` |
| Class method | `@classmethod` + `def method(cls):` | `cls` (the class) | `Class.method()` or `instance.method()` | no clean match — closest is a static factory method, but Java gives you no implicit class reference |
| Static method | `@staticmethod` + `def method():` | nothing extra | `Class.method()` or `instance.method()` | `static` method — direct match |

In practice, roughly 90% of the methods you write in everyday Python are instance methods (like everything in `TitanicCleaner` so far). `@staticmethod` shows up when you have a helper function that's logically related to the class but doesn't need object state. `@classmethod` shows up mainly for alternative constructors, and less commonly for tracking class-level state shared across all instances.