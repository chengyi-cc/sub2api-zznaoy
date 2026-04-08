export default {
  home: {
    viewOnGithub: 'Посмотреть на GitHub',
    viewDocs: 'Посмотреть документацию',
    docs: 'Документация',
    switchToLight: 'Переключить на светлую тему',
    switchToDark: 'Переключить на темную тему',
    dashboard: 'Панель',
    login: 'Войти',
    getStarted: 'Начать',
    goToDashboard: 'Перейти в панель',
    heroSubtitle: 'Один ключ, все модели ИИ',
    heroDescription: 'Не нужно управлять несколькими подписками. Доступ к Claude, GPT, Gemini и другим по одному API-ключу',
    tags: {
      subscriptionToApi: 'Подписка в API',
      stickySession: 'Сохранение сессий',
      realtimeBilling: 'Оплата по факту'
    },
    painPoints: {
      title: 'Знакомо?',
      items: {
        expensive: {
          title: 'Высокая стоимость подписок',
          desc: 'Оплата нескольких подписок на ИИ, которые суммарно обходятся дорого каждый месяц'
        },
        complex: {
          title: 'Хаос аккаунтов',
          desc: 'Управление разрозненными аккаунтами и API-ключами на разных платформах'
        },
        unstable: {
          title: 'Перебои в работе сервиса',
          desc: 'Один аккаунт упирается в лимиты и прерывает ваш рабочий процесс'
        },
        noControl: {
          title: 'Нет контроля использования',
          desc: 'Нельзя отследить, куда уходит бюджет, или ограничить использование участниками команды'
        }
      }
    },
    solutions: {
      title: 'Мы решаем эти проблемы',
      subtitle: 'Три простых шага к беспроблемному доступу к ИИ'
    },
    features: {
      unifiedGateway: 'Доступ в один клик',
      unifiedGatewayDesc: 'Получите один API-ключ для вызова всех подключенных моделей ИИ. Отдельные приложения не нужны.',
      multiAccount: 'Всегда надежно',
      multiAccountDesc: 'Умная маршрутизация между несколькими upstream-аккаунтами с автоматическим фейловером. Попрощайтесь с ошибками.',
      balanceQuota: 'Платите за использование',
      balanceQuotaDesc: 'Оплата по факту с лимитами квот. Полная прозрачность потребления команды.'
    },
    comparison: {
      title: 'Почему выбирают нас?',
      headers: {
        feature: 'Сравнение',
        official: 'Официальные подписки',
        us: 'Наша платформа'
      },
      items: {
        pricing: {
          feature: 'Цена',
          official: 'Фиксированная месячная плата, платите даже если не используете',
          us: 'Платите только за то, что используете'
        },
        models: {
          feature: 'Выбор моделей',
          official: 'Только один провайдер',
          us: 'Свободно переключайтесь между моделями'
        },
        management: {
          feature: 'Управление аккаунтами',
          official: 'Управляйте каждым сервисом отдельно',
          us: 'Единый ключ, одна панель'
        },
        stability: {
          feature: 'Стабильность',
          official: 'Лимиты одного аккаунта',
          us: 'Пул аккаунтов, автофейловер'
        },
        control: {
          feature: 'Контроль использования',
          official: 'Недоступно',
          us: 'Квоты и детальная аналитика'
        }
      }
    },
    providers: {
      title: 'Поддерживаемые модели ИИ',
      description: 'Один API, множество вариантов',
      supported: 'Поддерживается',
      soon: 'Скоро',
      claude: 'Claude',
      gemini: 'Gemini',
      antigravity: 'Antigravity',
      more: 'Еще'
    },
    cta: {
      title: 'Готовы начать?',
      description: 'Зарегистрируйтесь сейчас и получите бесплатные пробные кредиты, чтобы оценить бесшовный доступ к ИИ',
      button: 'Зарегистрироваться бесплатно'
    },
    footer: {
      allRightsReserved: 'Все права защищены.'
    }
  },
  keyUsage: {
    title: 'Использование API-ключа',
    subtitle: 'Введите ваш API-ключ, чтобы увидеть расходы и статус использования в реальном времени',
    placeholder: 'sk-ant-mirror-xxxxxxxxxxxx',
    query: 'Запросить',
    querying: 'Запрос выполняется...',
    privacyNote: 'Ваш ключ обрабатывается локально в браузере и не будет сохранен',
    dateRange: 'Диапазон дат:',
    dateRangeToday: 'Сегодня',
    dateRange7d: '7 дней',
    dateRange30d: '30 дней',
    dateRangeCustom: 'Пользовательский',
    apply: 'Применить',
    used: 'Использовано',
    detailInfo: 'Подробная информация',
    tokenStats: 'Статистика токенов',
    modelStats: 'Статистика использования моделей',
    model: 'Модель',
    requests: 'Запросы',
    inputTokens: 'Входные токены',
    outputTokens: 'Выходные токены',
    cacheCreationTokens: 'Создание кэша',
    cacheReadTokens: 'Чтение кэша',
    totalTokens: 'Всего токенов',
    cost: 'Стоимость',
    quotaMode: 'Режим квоты ключа',
    walletBalance: 'Баланс кошелька',
    totalQuota: 'Общая квота',
    limit5h: 'Лимит на 5 часов',
    limitDaily: 'Дневной лимит',
    limit7d: 'Лимит на 7 дней',
    limitWeekly: 'Недельный лимит',
    limitMonthly: 'Месячный лимит',
    remainingQuota: 'Остаток квоты',
    expiresAt: 'Истекает',
    todayExpires: '(истекает сегодня)',
    daysLeft: '({days} дней)',
    usedQuota: 'Использованная квота',
    resetNow: 'Скоро сброс',
    subscriptionType: 'Тип подписки',
    subscriptionExpires: 'Подписка истекает',
    todayRequests: 'Запросов сегодня',
    todayInputTokens: 'Вход сегодня',
    todayOutputTokens: 'Выход сегодня',
    todayTokens: 'Токенов сегодня',
    todayCacheCreation: 'Создание кэша сегодня',
    todayCacheRead: 'Чтение кэша сегодня',
    todayCost: 'Стоимость сегодня',
    rpmTpm: 'RPM / TPM',
    totalRequests: 'Всего запросов',
    totalInputTokens: 'Всего вход',
    totalOutputTokens: 'Всего выход',
    totalTokensLabel: 'Всего токенов',
    totalCacheCreation: 'Всего создание кэша',
    totalCacheRead: 'Всего чтение кэша',
    totalCost: 'Всего стоимость',
    avgDuration: 'Средняя длительность',
    enterApiKey: 'Пожалуйста, введите API-ключ',
    querySuccess: 'Запрос выполнен успешно',
    queryFailed: 'Запрос не выполнен',
    queryFailedRetry: 'Запрос не выполнен, пожалуйста, попробуйте позже'
  },
  setup: {
    title: 'Настройка Sub2API',
    description: 'Настройте ваш экземпляр Sub2API',
    database: {
      title: 'Настройка базы данных',
      description: 'Подключитесь к вашей базе данных PostgreSQL',
      host: 'Хост',
      port: 'Порт',
      username: 'Имя пользователя',
      password: 'Пароль',
      databaseName: 'Имя базы данных',
      sslMode: 'Режим SSL',
      passwordPlaceholder: 'Пароль',
      ssl: {
        disable: 'Отключить',
        require: 'Требовать',
        verifyCa: 'Проверять CA',
        verifyFull: 'Проверять полностью'
      }
    },
    redis: {
      title: 'Настройка Redis',
      description: 'Подключитесь к вашему серверу Redis',
      host: 'Хост',
      port: 'Порт',
      password: 'Пароль (необязательно)',
      database: 'База данных',
      passwordPlaceholder: 'Пароль',
      enableTls: 'Включить TLS',
      enableTlsHint: 'Использовать TLS при подключении к Redis (публичные CA-сертификаты)'
    },
    admin: {
      title: 'Администраторский аккаунт',
      description: 'Создайте аккаунт администратора',
      email: 'Email',
      password: 'Пароль',
      confirmPassword: 'Подтвердите пароль',
      passwordPlaceholder: 'Мин. 8 символов',
      confirmPasswordPlaceholder: 'Подтвердите пароль',
      passwordMismatch: 'Пароли не совпадают'
    },
    ready: {
      title: 'Готово к установке',
      description: 'Проверьте настройки и завершите установку',
      database: 'База данных',
      redis: 'Redis',
      adminEmail: 'Email администратора'
    },
    status: {
      testing: 'Проверка...',
      success: 'Подключение успешно',
      testConnection: 'Проверить подключение',
      installing: 'Установка...',
      completeInstallation: 'Завершить установку',
      completed: 'Установка завершена!',
      redirecting: 'Переходим на страницу входа...',
      restarting: 'Сервис перезапускается, подождите...',
      timeout: 'Перезапуск занимает больше времени, чем ожидалось. Пожалуйста, обновите страницу вручную.'
    }
  },
  common: {
    loading: 'Загрузка...',
    justNow: 'только что',
    save: 'Сохранить',
    cancel: 'Отмена',
    delete: 'Удалить',
    edit: 'Редактировать',
    create: 'Создать',
    update: 'Обновить',
    confirm: 'Подтвердить',
    reset: 'Сбросить',
    search: 'Поиск',
    filter: 'Фильтр',
    export: 'Экспорт',
    import: 'Импорт',
    actions: 'Действия',
    status: 'Статус',
    name: 'Имя',
    email: 'Email',
    password: 'Пароль',
    submit: 'Отправить',
    back: 'Назад',
    next: 'Далее',
    yes: 'Да',
    no: 'Нет',
    all: 'Все',
    none: 'Нет',
    noData: 'Нет данных',
    expand: 'Развернуть',
    collapse: 'Свернуть',
    success: 'Успешно',
    error: 'Ошибка',
    critical: 'Критично',
    warning: 'Предупреждение',
    info: 'Информация',
    active: 'Активно',
    inactive: 'Неактивно',
    more: 'Еще',
    close: 'Закрыть',
    enabled: 'Включено',
    disabled: 'Отключено',
    total: 'Итого',
    balance: 'Баланс',
    available: 'Доступно',
    copiedToClipboard: 'Скопировано в буфер обмена',
    copied: 'Скопировано',
    copyFailed: 'Не удалось скопировать',
    verifying: 'Проверка...',
    processing: 'Обработка...',
    contactSupport: 'Связаться с поддержкой',
    add: 'Добавить',
    invalidEmail: 'Введите корректный email адрес',
    optional: 'необязательно',
    selectOption: 'Выберите вариант',
    searchPlaceholder: 'Поиск...',
    noOptionsFound: 'Варианты не найдены',
    noGroupsAvailable: 'Нет доступных групп',
    unknownError: 'Произошла неизвестная ошибка',
    saving: 'Сохранение...',
    selectedCount: '({count} выбрано)',
    refresh: 'Обновить',
    settings: 'Настройки',
    chooseFile: 'Выбрать файл',
    notAvailable: 'Н/Д',
    now: 'Сейчас',
    unknown: 'Неизвестно',
    minutes: 'мин',
    time: {
      never: 'Никогда',
      justNow: 'Только что',
      minutesAgo: '{n}м назад',
      hoursAgo: '{n}ч назад',
      daysAgo: '{n}д назад',
