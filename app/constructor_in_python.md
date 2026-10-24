# `__init__`, `self`, and Attribute Declaration — Python vs Java

## `__init__` — yes, it's the constructor

In Java you write:

```java
public class TitanicCleaner {
    private String csvPath;
    private DataFrame df;

    public TitanicCleaner(String csvPath) {
        this.csvPath = csvPath;
        this.df = null;
    }
}
```

`__init__` is Python's equivalent of that constructor. It's a special method that runs automatically the moment you create an instance:

```python
cleaner = TitanicCleaner("titanic.csv")
```

That line calls `__init__` behind the scenes, with `csv_path="titanic.csv"`. The double underscores before and after (`__init__`) are Python's convention for "special/magic methods" that the language itself calls automatically — there's a whole family of these (`__str__`, `__len__`, `__eq__`, etc.) that you'll meet later. `__init__` specifically means "run this when an object is created."

One difference from Java: Python has no method *overloading*, so a class normally has exactly one `__init__`, not several constructors with different parameter lists. You handle optional parameters with default values instead (e.g. `def __init__(self, csv_path, encoding="utf-8")`).

## `self` — yes, it's like Java's `this`, with one big difference

In Java, `this` is implicit — you don't put it in the parameter list, it's just available inside instance methods. In Python, `self` is **explicit**: it's the first parameter of every instance method, and you have to write it yourself. Python doesn't guess which object you mean; it passes the object in as a real argument.

So when you call:

```python
cleaner = TitanicCleaner("titanic.csv")
```

Python effectively does:

```python
TitanicCleaner.__init__(cleaner, "titanic.csv")
```

`cleaner` (the object being built) gets passed in automatically as `self`. You never pass it yourself — Python does that for you whenever you call a method on an object — but you *do* have to declare it in the method signature. This trips up almost everyone coming from Java at first. Forgetting `self` as the first parameter, or forgetting `self.` when referring to an attribute inside a method, is probably the single most common beginner Python bug.

(Note: `self` is just a convention, not a keyword — you could technically name it anything — but literally everyone in the Python world uses `self`, so never deviate from it.)

## Why aren't `csv_path` and `df` declared inside `class TitanicCleaner:`?

This is the real conceptual difference from Java. In Java, you **must** declare fields at the class level before you can use them:

```java
public class TitanicCleaner {
    private String csvPath;   // must declare first
    private DataFrame df;     // must declare first
    ...
}
```

Python doesn't work that way. There's no separate "field declaration" step. An attribute comes into existence the moment you assign to it with `self.something = value` — typically the first time that happens is inside `__init__`, but it's not required to be:

```python
def __init__(self, csv_path):
    self.csv_path = csv_path   # <- this line IS the declaration AND the assignment
    self.df = None
```

The instant this line runs, Python creates an attribute called `csv_path` on that specific object and stores the value in it. There's no prior declaration anywhere in the class body — the assignment itself is what brings the attribute into being. This is a consequence of Python being **dynamically typed**: objects don't have a fixed, pre-declared shape the way Java objects do. Two instances of the same class could technically even end up with different attributes, though in well-written code you keep them consistent by always setting them in `__init__`.

That's also why `self.df = None` matters here even though we immediately overwrite it later in `load()`. It's a common Python pattern: declare the attribute in `__init__` with a placeholder value (`None`) so that:

1. Anyone reading the class can see at a glance what attributes an instance will have, without hunting through every method.
2. If some other method tries to use `self.df` before `load()` has been called, you get a clear `None`-related error rather than a confusing "attribute doesn't exist" error.

## Quick summary table

| Java | Python |
|---|---|
| Constructor: `public ClassName(args)` | `def __init__(self, args):` |
| `this` — implicit | `self` — explicit, must be first parameter |
| Fields declared at class level, then assigned in constructor | Attributes come into existence via `self.x = value`, usually first done in `__init__` |
| Fixed object shape, enforced by compiler | Dynamic — attributes exist because you assigned them, not because you declared a type |