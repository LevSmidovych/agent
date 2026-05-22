# Python decorators

Декоратор — це функція, яка приймає іншу функцію і повертає нову. Базовий шаблон:

```python
def my_decorator(fn):
    def wrapper(*args, **kwargs):
        # before
        result = fn(*args, **kwargs)
        # after
        return result
    return wrapper
```

Використання: `@my_decorator` над визначенням функції.
