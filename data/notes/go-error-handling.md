# Go error handling

Завжди перевіряй помилки:

```go
result, err := doSomething()
if err != nil {
    return fmt.Errorf("doing X: %w", err)
}
```

`%w` обгортає помилку, дозволяє `errors.Is()` і `errors.As()` у викликача.
