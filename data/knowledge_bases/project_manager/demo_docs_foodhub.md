# FoodHub — Внутрішня документація

## Процедура деплою

### Staging
1. Пушнути код у гілку `develop`
2. CI/CD автоматично збирає білд і деплоїть на staging.foodhub.ua
3. QA перевіряє протягом 1 робочого дня
4. Якщо критичних багів немає — мерж у `main`

### Production
1. Мерж `develop` → `main` створює реліз-тег автоматично
2. Backend деплоїться на AWS ECS (Fargate)
3. Мобільний додаток відправляється в App Store / Google Play
4. Rollback: повернутись до попереднього тегу командою `./rollback.sh v1.x.x`

### Контакти при аварії
- Андрій Бондаренко (backend): +380 67 XXX XX XX, Telegram @bondarenko_dev
- Марія Шевченко (frontend): +380 50 XXX XX XX
- DevOps: infra@foodhub.ua (зовнішній підрядник, SLA 4 години)

---

## API документація (основні ендпоінти)

### Авторизація
- `POST /api/auth/login` — логін по email + пароль, повертає JWT
- `POST /api/auth/register` — реєстрація нового користувача
- `POST /api/auth/refresh` — оновлення токена

### Замовлення
- `POST /api/orders` — створити замовлення (потрібен масив items, address, payment_method)
- `GET /api/orders/{id}` — статус замовлення (pending → confirmed → cooking → delivering → delivered)
- `PATCH /api/orders/{id}/cancel` — скасувати (тільки якщо статус pending або confirmed)

### Ресторани
- `GET /api/restaurants` — список з пагінацією, фільтри: cuisine, rating, delivery_time
- `GET /api/restaurants/{id}/menu` — меню ресторану з категоріями
- `POST /api/restaurants/{id}/reviews` — додати відгук (rating 1-5, text, order_id обов'язковий)

### Оплата
- Інтеграція з LiqPay
- Тестовий ключ: sandbox_xxxxxxxx (не використовувати в production!)
- Webhook для підтвердження: `POST /api/payments/callback`
- Підтримувані методи: card, google_pay, apple_pay

---

## Структура бази даних (основні таблиці)

### users
- id, email, phone, name, password_hash, role (customer/restaurant/admin), created_at

### restaurants  
- id, name, address, cuisine_type, rating, delivery_time_min, is_active, owner_id

### orders
- id, user_id, restaurant_id, status, total_amount, delivery_address, payment_method, payment_status, created_at, updated_at

### order_items
- id, order_id, menu_item_id, quantity, price

### reviews
- id, user_id, restaurant_id, order_id, rating, text, created_at

---

## Вимоги до безпеки
- Всі паролі зберігаються як bcrypt hash (cost factor 12)
- JWT токени: access 15 хвилин, refresh 30 днів
- Rate limiting: 100 запитів / хвилину на IP
- Дані кредитних карток НЕ зберігаються, тільки токен від LiqPay
- HTTPS обов'язковий, HTTP редиректить на HTTPS
- Логи зберігаються 90 днів, PII маскується

---

## Бізнес-правила

### Знижки та промокоди
- Промокод застосовується до загальної суми замовлення (без доставки)
- Максимальна знижка: 50%
- Мінімальна сума замовлення для промокоду: 200 грн
- Промокод WELCOME10 — 10% для нових користувачів (одноразовий)
- Промокод LUNCH20 — 20% на замовлення з 11:00 до 14:00 (будні)

### Доставка
- Безкоштовна доставка при замовленні від 500 грн
- Стандартна вартість доставки: 49 грн
- Максимальний радіус доставки: 10 км від ресторану
- Очікуваний час: delivery_time ресторану + 15 хвилин

### Скасування
- Безкоштовне скасування: статус pending або confirmed
- Якщо статус cooking або delivering — повернення 50% (вирішує менеджер)
- Автоматичне скасування якщо ресторан не підтвердив за 10 хвилин

---

## Метрики (за квітень 2026)
- Середній чек: 385 грн
- Замовлень на день: 120-150
- Середній час доставки: 42 хвилини
- Рейтинг у Google Play: 4.2 (87 відгуків)
- Рейтинг у App Store: 4.5 (23 відгуки)
- Конверсія (відкрив додаток → замовив): 12%
- Повторні замовлення: 35% протягом 30 днів
