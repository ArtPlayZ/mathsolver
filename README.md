# MathSolver

Калькулятор с текстовым интерфейсом: уравнения, неравенства, системы, графики.
Поддержка дробей, функций (sin/cos/sqrt/log), констант (pi, e).
Пошаговые решения. Sandbox-исполнитель с лимитами.

## Файлы
- mathservice.py — движок (токенайзер, парсер AST, решатель, графики, тесты).
- calculator.py — простой калькулятор.

## Запуск
    python mathservice.py eq "x^2-4=0"
    python mathservice.py ineq "x-2<0"
    python mathservice.py sys "x+y=3;x-y=1" x y
    python mathservice.py plot "sin(x)"

## Тесты
    python -m unittest mathservice -v

## Лицензия
MIT.
