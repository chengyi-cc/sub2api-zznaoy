      todayRequests: 'Запросы сегодня',
      newUsersToday: 'Новые пользователи сегодня',
      todayTokens: 'Токены сегодня',
      totalTokens: 'Всего токенов',
      cacheToday: 'Кэш (сегодня)',
      performance: 'Производительность',
      avgResponse: 'Ср. отклик',
      active: 'активно',
      ok: 'ок',
      err: 'ош',
      activeUsers: 'активные пользователи',
      create: 'Создать',
      timeRange: 'Диапазон времени',
      granularity: 'Детализация',
      day: 'День',
      hour: 'Час',
      modelDistribution: 'Распределение моделей',
      groupDistribution: 'Распределение использования по группам',
      metricTokens: 'По токенам',
      metricActualCost: 'По фактической стоимости',
      tokenUsageTrend: 'Динамика использования токенов',
      userUsageTrend: 'Динамика использования пользователями (Top 12)',
      model: 'Модель',
      group: 'Группа',
      noGroup: 'Без группы',
      requests: 'Запросы',
      tokens: 'Токены',
      actual: 'Факт',
      standard: 'Станд.',
      noDataAvailable: 'Нет данных',
      recentUsage: 'Недавнее использование',
      viewModelDistribution: 'Распределение моделей',
      viewSpendingRanking: 'Рейтинг расходов пользователей',
      spendingRankingTitle: 'Рейтинг расходов пользователей',
      spendingRankingUser: 'Пользователь',
      spendingRankingRequests: 'Запросы',
      spendingRankingTokens: 'Токены',
      spendingRankingSpend: 'Расход',
      spendingRankingOther: 'Другие',
      spendingRankingUsage: 'Использование',
      spendShort: 'Расх.',
      requestsShort: 'Запр.',
      tokensShort: 'Ток.',
      failedToLoad: 'Не удалось загрузить статистику панели'
    },
    backup: {
      title: 'Резервное копирование базы данных',
      description: 'Полное резервное копирование базы данных в S3-совместимое хранилище с плановым бэкапом и восстановлением',
      s3: {
        title: 'Конфигурация S3-хранилища',
        description: 'Настройка S3-совместимого хранилища (поддерживается Cloudflare R2)',
        descriptionPrefix: 'Настройка S3-совместимого хранилища (поддерживается',
        descriptionSuffix: ')',
        enabled: 'Включить S3-хранилище',
        endpoint: 'Endpoint',
        region: 'Регион',
        bucket: 'Bucket',
        prefix: 'Префикс ключа',
        accessKeyId: 'Access Key ID',
        secretAccessKey: 'Secret Access Key',
        secretConfigured: 'Уже настроено, оставьте пустым, чтобы сохранить',
        forcePathStyle: 'Принудительный Path Style',
        testConnection: 'Проверить соединение',
        testSuccess: 'Проверка соединения с S3 прошла успешно',
        testFailed: 'Не удалось проверить соединение с S3',
        saved: 'Конфигурация S3 сохранена'
      },
      schedule: {
        title: 'Плановое резервное копирование',
        description: 'Настройка автоматического резервного копирования по расписанию',
        enabled: 'Включить плановое резервное копирование',
        cronExpr: 'Cron-выражение',
        cronHint: 'например, "0 2 * * *" означает каждый день в 02:00',
        retainDays: 'Срок хранения бэкапов (дни)',
        retainDaysHint: 'Файлы резервных копий будут автоматически удаляться через указанное число дней, 0 = не истекают',
        retainCount: 'Максимум сохраняемых копий',
        retainCountHint: 'Максимальное количество хранимых резервных копий, 0 = без ограничений',
        saved: 'Конфигурация расписания сохранена'
      },
      operations: {
        title: 'Записи резервного копирования',
        description: 'Создание ручных резервных копий и управление существующими записями',
        createBackup: 'Создать резервную копию',
        backing: 'Создание резервной копии...',
        backupCreated: 'Резервная копия успешно создана',
        expireDays: 'Срок действия (дни)',
        alreadyInProgress: 'Резервное копирование уже выполняется',
        backupRunning: 'Резервное копирование выполняется...',
        backupFailed: 'Не удалось создать резервную копию',
        restoreRunning: 'Восстановление выполняется...',
        restoreFailed: 'Не удалось выполнить восстановление'
      },
      columns: {
        status: 'Статус',
        fileName: 'Имя файла',
        size: 'Размер',
        expiresAt: 'Истекает',
        triggeredBy: 'Запущено',
        startedAt: 'Начато',
        actions: 'Действия'
      },
      status: {
        pending: 'В ожидании',
        running: 'Выполняется',
        completed: 'Завершено',
        failed: 'Ошибка'
      },
      progress: {
        pending: 'Подготовка',
        dumping: 'Экспорт базы данных',
        uploading: 'Загрузка'
      },
      trigger: {
        manual: 'Вручную',
        scheduled: 'По расписанию'
      },
      neverExpire: 'Никогда',
      empty: 'Нет записей резервного копирования',
      actions: {
        download: 'Скачать',
        restore: 'Восстановить',
        restoreConfirm: 'Вы уверены, что хотите восстановить данные из этой резервной копии? Текущая база данных будет перезаписана!',
        restorePasswordPrompt: 'Введите пароль администратора для подтверждения восстановления',
        restoreSuccess: 'База данных успешно восстановлена',
        deleteConfirm: 'Вы уверены, что хотите удалить эту резервную копию?',
        deleted: 'Резервная копия удалена'
      },
      r2Guide: {
        title: 'Руководство по настройке Cloudflare R2',
        intro: 'Cloudflare R2 предоставляет S3-совместимое объектное хранилище с бесплатным лимитом 10 ГБ + 1 млн запросов Class A в месяц, что отлично подходит для резервного копирования базы данных.',
        step1: {
          title: 'Создайте bucket R2',
          line1: 'Войдите в Cloudflare Dashboard (dash.cloudflare.com) и выберите "R2 Object Storage" в боковом меню',
          line2: 'Нажмите "Create bucket", введите имя (например, sub2api-backups) и выберите регион',
          line3: 'Нажмите Create для завершения'
        },
        step2: {
          title: 'Создайте API-токен',
          line1: 'На странице R2 нажмите "Manage R2 API Tokens" в правом верхнем углу',
          line2: 'Нажмите "Create API token" и установите права "Object Read & Write"',
          line3: 'Рекомендуется ограничить токен конкретным bucket для лучшей безопасности',
          line4: 'После создания вы увидите Access Key ID и Secret Access Key',
          warning: 'Secret Access Key показывается только один раз, сразу скопируйте и сохраните его!'
        },
        step3: {
          title: 'Получите S3 Endpoint',
          desc: 'Найдите ваш Account ID на обзорной странице R2 (в URL или в правой панели). Формат endpoint такой:',
          accountId: 'your_account_id'
        },
        step4: {
          title: 'Заполните конфигурацию',
          checkEnabled: 'Отмечено',
          bucketValue: 'Имя вашего bucket',
          fromStep2: 'Значение из шага 2',
          unchecked: 'Не отмечено'
        },
        freeTier: 'Бесплатный тариф R2: 10 ГБ хранилища + 1 млн запросов Class A + 10 млн запросов Class B в месяц, этого более чем достаточно для резервного копирования базы данных.'
      }
    },
    dataManagement: {
      title: 'Управление данными',
      description: 'Управляйте состоянием агента управления данными, настройками объектного хранилища и заданиями резервного копирования в одном месте',
      agent: {
        title: 'Состояние агента управления данными',
        description: 'Система проверяет фиксированный Unix-сокет и включает функции управления данными только при доступности агента.',
        enabled: 'Агент управления данными готов. Операции управления данными доступны.',
        disabled: 'Агент управления данными недоступен. Сейчас доступна только диагностическая информация.',
        socketPath: 'Путь к сокету',
        version: 'Версия',
        status: 'Статус',
        uptime: 'Время работы',
        reasonLabel: 'Причина недоступности',
        reason: {
          DATA_MANAGEMENT_AGENT_SOCKET_MISSING: 'Файл сокета управления данными отсутствует',
          DATA_MANAGEMENT_AGENT_UNAVAILABLE: 'Агент управления данными недоступен',
          BACKUP_AGENT_SOCKET_MISSING: 'Файл сокета бэкап-агента отсутствует',
          BACKUP_AGENT_UNAVAILABLE: 'Бэкап-агент недоступен',
          UNKNOWN: 'Неизвестная причина'
        }
      },
      sections: {
        config: {
          title: 'Конфигурация резервного копирования',
          description: 'Настройка источника резервного копирования, политики хранения и параметров S3.'
        },
        s3: {
          title: 'Объектное хранилище S3',
          description: 'Настройка и проверка загрузки артефактов резервного копирования в стандартное S3-совместимое хранилище.'
        },
        backup: {
          title: 'Операции резервного копирования',
          description: 'Запуск заданий резервного копирования PostgreSQL, Redis и полного бэкапа.'
        },
        history: {
          title: 'История резервного копирования',
          description: 'Просмотр статуса заданий резервного копирования, ошибок и метаданных артефактов.'
        }
      },
      form: {
        sourceMode: 'Режим источника',
        backupRoot: 'Корень резервных копий',
        activePostgresProfile: 'Активный профиль PostgreSQL',
        activeRedisProfile: 'Активный профиль Redis',
        activeS3Profile: 'Активный профиль S3',
        retentionDays: 'Срок хранения (дни)',
        keepLast: 'Сохранять последние задания',
        uploadToS3: 'Загружать в S3',
        useActivePostgresProfile: 'Использовать активный профиль PostgreSQL',
        useActiveRedisProfile: 'Использовать активный профиль Redis',
        useActiveS3Profile: 'Использовать активный профиль',
        idempotencyKey: 'Ключ идемпотентности (необязательно)',
        secretConfigured: 'Уже настроено, оставьте пустым, чтобы сохранить без изменений',
        source: {
          profileID: 'ID профиля (уникальный)',
          profileName: 'Имя профиля',
          setActive: 'Сделать активным после создания'
        },
        postgres: {
          title: 'PostgreSQL',
          host: 'Хост',
          port: 'Порт',
          user: 'Пользователь',
          password: 'Пароль',
          database: 'База данных',
          sslMode: 'Режим SSL',
          containerName: 'Имя контейнера (режим docker_exec)'
        },
        redis: {
          title: 'Redis',
          addr: 'Адрес (host:port)',
          username: 'Имя пользователя',
          password: 'Пароль',
          db: 'Индекс базы данных',
          containerName: 'Имя контейнера (режим docker_exec)'
        },
        s3: {
          enabled: 'Включить загрузку в S3',
          profileID: 'ID профиля (уникальный)',
          profileName: 'Имя профиля',
          endpoint: 'Endpoint (необязательно)',
          region: 'Регион',
          bucket: 'Bucket',
          accessKeyID: 'Access Key ID',
          secretAccessKey: 'Secret Access Key',
          prefix: 'Префикс объекта',
          forcePathStyle: 'Принудительный Path Style',
          useSSL: 'Использовать SSL',
          setActive: 'Сделать активным после создания'
        }
      },
      sourceProfiles: {
        createTitle: 'Создать профиль источника',
        editTitle: 'Редактировать профиль источника',
        empty: 'Пока нет профилей источника, сначала создайте один',
        deleteConfirm: 'Удалить профиль источника {profileID}?',
        columns: {
          profile: 'Профиль',
          active: 'Активен',
          connection: 'Подключение',
          database: 'База данных',
          updatedAt: 'Обновлено',
          actions: 'Действия'
        }
      },
      s3Profiles: {
        createTitle: 'Создать профиль S3',
        editTitle: 'Редактировать профиль S3',
        empty: 'Пока нет профилей S3, сначала создайте один',
        editHint: 'Нажмите "Edit", чтобы изменить детали профиля в правой панели.',
        deleteConfirm: 'Удалить профиль S3 {profileID}?',
        columns: {
          profile: 'Профиль',
          active: 'Активен',
          storage: 'Хранилище',
          updatedAt: 'Обновлено',
          actions: 'Действия'
        }
      },
      history: {
        total: '{count} задач',
        empty: 'Пока нет задач резервного копирования',
        columns: {
          jobID: 'ID задачи',
          type: 'Тип',
          status: 'Статус',
          triggeredBy: 'Запущено',
          pgProfile: 'Профиль PostgreSQL',
          redisProfile: 'Профиль Redis',
          s3Profile: 'Профиль S3',
          finishedAt: 'Завершено',
          artifact: 'Артефакт',
          error: 'Ошибка'
        },
        status: {
          queued: 'В очереди',
          running: 'Выполняется',
          succeeded: 'Успешно',
          failed: 'Ошибка',
          partial_succeeded: 'Частично успешно'
        }
