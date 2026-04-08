        enabled: 'Включить это правило'
      },
      nameRequired: 'Пожалуйста, введите имя правила',
      conditionsRequired: 'Пожалуйста, настройте хотя бы один код ошибки или ключевое слово',
      ruleCreated: 'Правило успешно создано',
      ruleUpdated: 'Правило успешно обновлено',
      ruleDeleted: 'Правило успешно удалено',
      deleteConfirm: 'Вы уверены, что хотите удалить правило "{name}"?',
      failedToLoad: 'Не удалось загрузить правила',
      failedToSave: 'Не удалось сохранить правило',
      failedToDelete: 'Не удалось удалить правило',
      failedToToggle: 'Не удалось переключить статус'
    },
    tlsFingerprintProfiles: {
      title: 'Профили TLS fingerprint',
      description: 'Управление профилями TLS fingerprint для имитации характеристик TLS-handshake конкретных клиентов',
      createProfile: 'Создать профиль',
      editProfile: 'Редактировать профиль',
      deleteProfile: 'Удалить профиль',
      noProfiles: 'Профили не настроены',
      createFirstProfile: 'Создайте первый профиль TLS fingerprint',
      columns: {
        name: 'Имя',
        description: 'Описание',
        grease: 'GREASE',
        alpn: 'ALPN',
        actions: 'Действия'
      },
      form: {
        pasteYaml: 'Вставить YAML-конфигурацию',
        pasteYamlPlaceholder: 'Вставьте сюда YAML-вывод из TLS Fingerprint Collector...',
        pasteYamlHint: 'Вставьте YAML, скопированный из TLS Fingerprint Collector, чтобы автоматически заполнить все поля.',
        openCollector: 'Открыть Collector',
        parseYaml: 'Разобрать YAML',
        yamlParsed: 'YAML успешно разобран, поля заполнены автоматически',
        yamlParseFailed: 'Не удалось разобрать YAML: поле name не найдено',
        name: 'Имя профиля',
        namePlaceholder: 'например, macOS Node.js v24',
        description: 'Описание',
        descriptionPlaceholder: 'Необязательное описание профиля',
        enableGrease: 'Включить GREASE',
        enableGreaseHint: 'Вставлять значения GREASE в расширения TLS ClientHello',
        cipherSuites: 'Cipher Suites',
        cipherSuitesHint: 'Шестнадцатеричные значения через запятую, например 0x1301, 0x1302, 0xc02c',
        curves: 'Эллиптические кривые',
        curvesHint: 'ID кривых через запятую',
        pointFormats: 'Форматы точек',
        signatureAlgorithms: 'Алгоритмы подписи',
        alpnProtocols: 'ALPN-протоколы',
        alpnProtocolsHint: 'Через запятую, например h2, http/1.1',
        supportedVersions: 'Поддерживаемые версии TLS',
        keyShareGroups: 'Группы Key Share',
        pskModes: 'Режимы PSK',
        extensions: 'Расширения'
      },
      deleteConfirm: 'Удалить профиль',
      deleteConfirmMessage: 'Вы уверены, что хотите удалить профиль "{name}"? Аккаунты, использующие этот профиль, переключатся на встроенное значение по умолчанию.',
      createSuccess: 'Профиль успешно создан',
      updateSuccess: 'Профиль успешно обновлён',
      deleteSuccess: 'Профиль успешно удалён',
      loadFailed: 'Не удалось загрузить профили',
      saveFailed: 'Не удалось сохранить профиль',
      deleteFailed: 'Не удалось удалить профиль'
    }
  },
  subscriptionProgress: {
    title: 'Мои подписки',
    viewDetails: 'Просмотреть детали подписки',
    activeCount: 'Активных подписок: {count}',
    daily: 'Ежедневно',
    weekly: 'Еженедельно',
    monthly: 'Ежемесячно',
    daysRemaining: 'Осталось {days} дн.',
    expired: 'Истекла',
    expiresToday: 'Истекает сегодня',
    expiresTomorrow: 'Истекает завтра',
    viewAll: 'Просмотреть все подписки',
    noSubscriptions: 'Нет активных подписок',
    unlimited: 'Без ограничений'
  },
  version: {
    currentVersion: 'Текущая версия',
    latestVersion: 'Последняя версия',
    upToDate: 'У вас установлена последняя версия.',
    updateAvailable: 'Доступна новая версия!',
    releaseNotes: 'Примечания к релизу',
    noReleaseNotes: 'Нет примечаний к релизу',
    viewUpdate: 'Просмотреть обновление',
    viewRelease: 'Просмотреть релиз',
    viewChangelog: 'Просмотреть changelog',
    refresh: 'Обновить',
    sourceMode: 'Сборка из исходников',
    sourceModeHint: 'Сборка из исходников, для обновления используйте git pull',
    updateNow: 'Обновить сейчас',
    updating: 'Обновление...',
    updateComplete: 'Обновление завершено',
    updateFailed: 'Обновление не удалось',
    restartRequired: 'Пожалуйста, перезапустите сервис, чтобы применить обновление',
    restartNow: 'Перезапустить сейчас',
    restarting: 'Перезапуск...',
    retry: 'Повторить'
  },
  purchase: {
    title: 'Пополнение / Подписка',
    description: 'Пополнение баланса или покупка подписки через встроенную страницу',
    openInNewTab: 'Открыть в новой вкладке',
    notEnabledTitle: 'Функция не включена',
    notEnabledDesc: 'Администратор не включил пункт пополнения/подписки. Пожалуйста, свяжитесь с администратором.',
    notConfiguredTitle: 'URL пополнения / подписки не настроен',
    notConfiguredDesc: 'Администратор включил этот пункт, но не настроил URL пополнения/подписки. Пожалуйста, свяжитесь с администратором.'
  },
  customPage: {
    title: 'Пользовательская страница',
    openInNewTab: 'Открыть в новой вкладке',
    notFoundTitle: 'Страница не найдена',
    notFoundDesc: 'Эта пользовательская страница не существует или была удалена.',
    notConfiguredTitle: 'URL страницы не настроен',
    notConfiguredDesc: 'URL этой пользовательской страницы настроен некорректно.'
  },
  announcements: {
    title: 'Объявления',
    description: 'Просмотр системных объявлений',
    unreadOnly: 'Показывать только непрочитанные',
    markRead: 'Отметить как прочитанное',
    markAllRead: 'Отметить всё как прочитанное',
    viewAll: 'Просмотреть все объявления',
    markedAsRead: 'Отмечено как прочитанное',
    allMarkedAsRead: 'Все объявления отмечены как прочитанные',
    newCount: '{count} новое объявление | {count} новых объявлений',
    readAt: 'Прочитано',
    read: 'Прочитано',
    unread: 'Не прочитано',
    startsAt: 'Начинается',
    endsAt: 'Заканчивается',
    empty: 'Объявлений нет',
    emptyUnread: 'Нет непрочитанных объявлений',
    total: 'объявлений',
    emptyDescription: 'Сейчас системных объявлений нет',
    readStatus: 'Вы уже прочитали это объявление',
    markReadHint: 'Нажмите "Отметить как прочитанное", чтобы отметить это объявление'
  },
  userSubscriptions: {
    title: 'Мои подписки',
    description: 'Просмотр ваших тарифных планов и использования',
    noActiveSubscriptions: 'Нет активных подписок',
    noActiveSubscriptionsDesc: 'У вас нет активных подписок. Обратитесь к администратору, чтобы получить подписку.',
    failedToLoad: 'Не удалось загрузить подписки',
    status: {
      active: 'Активна',
      expired: 'Истекла',
      revoked: 'Отозвана'
    },
    usage: 'Использование',
    expires: 'Истекает',
    noExpiration: 'Без срока действия',
    unlimited: 'Без ограничений',
    unlimitedDesc: 'Для этой подписки нет лимитов использования',
    daily: 'Ежедневно',
    weekly: 'Еженедельно',
    monthly: 'Ежемесячно',
    daysRemaining: 'Осталось {days} дн.',
    expiresOn: 'Истекает {date}',
    resetIn: 'Сброс через {time}',
    windowNotActive: 'Ожидает первого использования',
    usageOf: '{used} из {limit}'
  },
  onboarding: {
    restartTour: 'Перезапустить onboarding-тур',
    dontShowAgain: 'Больше не показывать',
    dontShowAgainTitle: 'Навсегда закрыть onboarding-руководство',
    confirmDontShow: 'Вы уверены, что больше не хотите видеть onboarding-руководство?\n\nВы сможете перезапустить его в любой момент из пользовательского меню в правом верхнем углу.',
    confirmExit: 'Вы уверены, что хотите выйти из onboarding-руководства? Его можно перезапустить в любой момент из меню в правом верхнем углу.',
    interactiveHint: 'Нажмите Enter или кликните, чтобы продолжить',
    navigation: {
      flipPage: 'Следующая страница',
      exit: 'Выход'
    },
    admin: {
      welcome: {
        title: 'Добро пожаловать в Sub2API',
        description: '<div style="line-height: 1.8;"><p style="margin-bottom: 16px;">Sub2API — это мощная платформа-шлюз для AI-сервисов, которая помогает легко управлять и распределять доступ к AI.</p><p style="margin-bottom: 12px;"><b>Ключевые возможности:</b></p><ul style="margin-left: 20px; margin-bottom: 16px;"><li><b>Управление группами</b> — создание тарифных уровней (VIP, пробный доступ и т.д.)</li><li><b>Пул аккаунтов</b> — подключение нескольких upstream-аккаунтов AI-сервисов</li><li><b>Распределение ключей</b> — выдача независимых API-ключей пользователям</li><li><b>Контроль биллинга</b> — гибкое управление тарифами и квотами</li></ul><p style="color: #10b981; font-weight: 600;">Давайте завершим базовую настройку примерно за 3 минуты.</p></div>',
        nextBtn: 'Начать настройку',
        prevBtn: 'Пропустить'
      },
      groupManage: {
        title: 'Шаг 1: Управление группами',
        description: '<div style="line-height: 1.7;"><p style="margin-bottom: 12px;"><b>Что такое группа?</b></p><p style="margin-bottom: 12px;">Группы — это ключевая сущность Sub2API, своего рода "пакет сервиса":</p><ul style="margin-left: 20px; margin-bottom: 12px; font-size: 13px;"><li>В каждой группе может быть несколько upstream-аккаунтов</li><li>У каждой группы свой коэффициент биллинга</li><li>Группа может быть публичной или эксклюзивной</li></ul><p style="margin-top: 12px; padding: 8px 12px; background: #f0fdf4; border-left: 3px solid #10b981; border-radius: 4px; font-size: 13px;"><b>Пример:</b> можно создать группы "VIP Premium" (высокий тариф) и "Free Trial" (низкий тариф)</p><p style="margin-top: 16px; color: #10b981; font-weight: 600;">Нажмите "Управление группами" в левом меню</p></div>'
      },
      createGroup: {
        title: 'Создание новой группы',
        description: '<div style="line-height: 1.7;"><p style="margin-bottom: 12px;">Давайте создадим вашу первую группу.</p><p style="padding: 8px 12px; background: #eff6ff; border-left: 3px solid #3b82f6; border-radius: 4px; font-size: 13px; margin-bottom: 12px;"><b>Совет:</b> сначала создайте тестовую группу, чтобы познакомиться с процессом</p><p style="color: #10b981; font-weight: 600;">Нажмите кнопку "Создать группу"</p></div>'
      },
      groupName: {
        title: '1. Название группы',
        description: '<div style="line-height: 1.7;"><p style="margin-bottom: 12px;">Дайте группе понятное и легко узнаваемое имя.</p><div style="padding: 8px 12px; background: #f0fdf4; border-left: 3px solid #10b981; border-radius: 4px; font-size: 13px; margin-bottom: 12px;"><b>Советы по именованию:</b><ul style="margin: 8px 0 0 16px;"><li>"Test Group" — для тестирования</li><li>"VIP Premium" — для качественного сервиса</li><li>"Free Trial" — для пробного доступа</li></ul></div><p style="font-size: 13px; color: #6b7280;">После заполнения нажмите "Далее"</p></div>',
        nextBtn: 'Далее'
      },
      groupPlatform: {
        title: '2. Выбор платформы',
        description: '<div style="line-height: 1.7;"><p style="margin-bottom: 12px;">Выберите AI-платформу, которую будет поддерживать эта группа.</p><div style="padding: 8px 12px; background: #eff6ff; border-left: 3px solid #3b82f6; border-radius: 4px; font-size: 13px; margin-bottom: 12px;"><b>Подсказка по платформам:</b><ul style="margin: 8px 0 0 16px;"><li><b>Anthropic</b> — модели Claude</li><li><b>OpenAI</b> — модели GPT</li><li><b>Google</b> — модели Gemini</li></ul></div><p style="font-size: 13px; color: #6b7280;">Одна группа может относиться только к одной платформе</p></div>',
        nextBtn: 'Далее'
      },
      groupMultiplier: {
        title: '3. Тарифный коэффициент',
        description: '<div style="line-height: 1.7;"><p style="margin-bottom: 12px;">Настройте коэффициент биллинга, чтобы управлять стоимостью для пользователей.</p><div style="padding: 8px 12px; background: #fef3c7; border-left: 3px solid #f59e0b; border-radius: 4px; font-size: 13px; margin-bottom: 12px;"><b>Правила биллинга:</b><ul style="margin: 8px 0 0 16px;"><li><b>1.0</b> — исходная цена (себестоимость)</li><li><b>1.5</b> — пользователь потребил на $1, будет списано $1.5</li><li><b>2.0</b> — пользователь потребил на $1, будет списано $2</li><li><b>0.8</b> — режим субсидии (убыточный)</li></ul></div><p style="font-size: 13px; color: #6b7280;">Для тестовой группы рекомендуется установить 1.0</p></div>',
        nextBtn: 'Далее'
      },
      groupExclusive: {
        title: '4. Эксклюзивная группа (необязательно)',
        description: '<div style="line-height: 1.7;"><p style="margin-bottom: 12px;">Управляйте видимостью группы и правами доступа.</p><div style="padding: 8px 12px; background: #eff6ff; border-left: 3px solid #3b82f6; border-radius: 4px; font-size: 13px; margin-bottom: 12px;"><b>Пояснение:</b><ul style="margin: 8px 0 0 16px;"><li><b>Выключено</b> — публичная группа, видна всем пользователям</li><li><b>Включено</b> — эксклюзивная группа, доступна только указанным пользователям</li></ul></div><p style="padding: 8px 12px; background: #f0fdf4; border-left: 3px solid #10b981; border-radius: 4px; font-size: 13px;"><b>Сценарии использования:</b> VIP-доступ, внутреннее тестирование, специальные клиенты</p></div>',
        nextBtn: 'Далее'
      },
      groupSubmit: {
        title: 'Сохранение группы',
        description: '<div style="line-height: 1.7;"><p style="margin-bottom: 12px;">Проверьте информацию и нажмите создать, чтобы сохранить группу.</p><p style="padding: 8px 12px; background: #fef3c7; border-left: 3px solid #f59e0b; border-radius: 4px; font-size: 13px; margin-bottom: 12px;"><b>Важно:</b> тип платформы нельзя изменить после создания, но остальные параметры можно редактировать в любой момент</p><p style="padding: 8px 12px; background: #f0fdf4; border-left: 3px solid #10b981; border-radius: 4px; font-size: 13px;"><b>Следующий шаг:</b> после создания мы добавим в группу upstream-аккаунты</p><p style="margin-top: 12px; color: #10b981; font-weight: 600;">Нажмите кнопку "Создать"</p></div>'
      },
      accountManage: {
        title: 'Шаг 2: Добавление аккаунта',
        description: '<div style="line-height: 1.7;"><p style="margin-bottom: 12px;"><b>Отлично! Группа успешно создана.</b></p><p style="margin-bottom: 12px;">Теперь добавьте upstream-аккаунты AI-сервисов, чтобы сервис действительно начал работать.</p><div style="padding: 8px 12px; background: #eff6ff; border-left: 3px solid #3b82f6; border-radius: 4px; font-size: 13px; margin-bottom: 12px;"><b>Зачем нужны аккаунты:</b><ul style="margin: 8px 0 0 16px;"><li>Подключение к upstream AI-сервисам (Claude, GPT и т.д.)</li><li>В одной группе может быть несколько аккаунтов для балансировки нагрузки</li><li>Поддерживаются OAuth и Session Key</li></ul></div><p style="margin-top: 16px; color: #10b981; font-weight: 600;">Нажмите "Управление аккаунтами" в левом меню</p></div>'
      },
      createAccount: {
        title: 'Добавление нового аккаунта',
        description: '<div style="line-height: 1.7;"><p style="margin-bottom: 12px;">Нажмите кнопку, чтобы добавить первый upstream-аккаунт.</p><p style="padding: 8px 12px; background: #f0fdf4; border-left: 3px solid #10b981; border-radius: 4px; font-size: 13px;"><b>Совет:</b> рекомендуется использовать OAuth — это безопаснее и не требует ручного извлечения ключей</p><p style="margin-top: 12px; color: #10b981; font-weight: 600;">Нажмите кнопку "Добавить аккаунт"</p></div>'
      },
      accountName: {
        title: '1. Имя аккаунта',
        description: '<div style="line-height: 1.7;"><p style="margin-bottom: 12px;">Укажите понятное имя для аккаунта.</p><p style="padding: 8px 12px; background: #f0fdf4; border-left: 3px solid #10b981; border-radius: 4px; font-size: 13px;"><b>Советы по именованию:</b> "Claude Main", "GPT Backup 1", "Test Account" и т.д.</p></div>',
        nextBtn: 'Далее'
      },
      accountPlatform: {
        title: '2. Выбор платформы',
        description: '<div style="line-height: 1.7;"><p style="margin-bottom: 12px;">Выберите платформу-провайдера для этого аккаунта.</p><p style="padding: 8px 12px; background: #fef3c7; border-left: 3px solid #f59e0b; border-radius: 4px; font-size: 13px;"><b>Важно:</b> платформа должна совпадать с платформой только что созданной группы</p></div>',
        nextBtn: 'Далее'
      },
      accountType: {
        title: '3. Способ авторизации',
        description: '<div style="line-height: 1.7;"><p style="margin-bottom: 12px;">Выберите способ авторизации аккаунта.</p><div style="padding: 8px 12px; background: #f0fdf4; border-left: 3px solid #10b981; border-radius: 4px; font-size: 13px; margin-bottom: 12px;"><b>Рекомендуется: OAuth</b><ul style="margin: 8px 0 0 16px;"><li>Не требует ручного извлечения ключей</li><li>Более безопасен и поддерживает автообновление</li><li>Работает с Claude Code, ChatGPT OAuth</li></ul></div><div style="padding: 8px 12px; background: #eff6ff; border-left: 3px solid #3b82f6; border-radius: 4px; font-size: 13px;"><b>Способ Session Key</b><ul style="margin: 8px 0 0 16px;"><li>Требует ручного извлечения из браузера</li><li>Может требовать периодического обновления</li><li>Подходит для платформ без OAuth</li></ul></div></div>',
        nextBtn: 'Далее'
      },
      accountPriority: {
        title: '4. Приоритет (необязательно)',
        description: '<div style="line-height: 1.7;"><p style="margin-bottom: 12px;">Настройте приоритет вызовов аккаунта.</p><div style="padding: 8px 12px; background: #eff6ff; border-left: 3px solid #3b82f6; border-radius: 4px; font-size: 13px; margin-bottom: 12px;"><b>Правила приоритета:</b><ul style="margin: 8px 0 0 16px;"><li>Чем меньше число, тем выше приоритет</li><li>Система сначала использует аккаунты с меньшим значением</li><li>При одинаковом приоритете выбор случайный</li></ul></div><p style="padding: 8px 12px; background: #f0fdf4; border-left: 3px solid #10b981; border-radius: 4px; font-size: 13px;"><b>Пример:</b> для основного аккаунта задайте более низкое значение, для резервных — более высокое</p></div>',
        nextBtn: 'Далее'
      },
      accountGroups: {
        title: '5. Назначение групп',
        description: '<div style="line-height: 1.7;"><p style="margin-bottom: 12px;"><b>Ключевой шаг!</b> Назначьте аккаунт группе, которую вы только что создали.</p><div style="padding: 8px 12px; background: #fee2e2; border-left: 3px solid #ef4444; border-radius: 4px; font-size: 13px; margin-bottom: 12px;"><b>Важное напоминание:</b><ul style="margin: 8px 0 0 16px;"><li>Нужно выбрать хотя бы одну группу</li><li>Неназначенные аккаунты нельзя использовать</li><li>Один аккаунт можно назначить нескольким группам</li></ul></div><p style="padding: 8px 12px; background: #f0fdf4; border-left: 3px solid #10b981; border-radius: 4px; font-size: 13px;"><b>Совет:</b> выберите тестовую группу, которую вы только что создали</p></div>',
        nextBtn: 'Далее'
      },
      accountSubmit: {
        title: 'Сохранение аккаунта',
        description: '<div style="line-height: 1.7;"><p style="margin-bottom: 12px;">Проверьте информацию и нажмите сохранить.</p><div style="padding: 8px 12px; background: #eff6ff; border-left: 3px solid #3b82f6; border-radius: 4px; font-size: 13px; margin-bottom: 12px;"><b>Поток OAuth:</b><ul style="margin: 8px 0 0 16px;"><li>После нажатия "Сохранить" будет открыта страница провайдера</li><li>Завершите вход и авторизацию на стороне провайдера</li><li>После успешной авторизации произойдёт автоматический возврат</li></ul></div><p style="padding: 8px 12px; background: #f0fdf4; border-left: 3px solid #10b981; border-radius: 4px; font-size: 13px;"><b>Следующий шаг:</b> после добавления аккаунта мы создадим API-ключ</p><p style="margin-top: 12px; color: #10b981; font-weight: 600;">Нажмите кнопку "Сохранить"</p></div>'
      },
      keyManage: {
        title: 'Шаг 3: Генерация ключа',
        description: '<div style="line-height: 1.7;"><p style="margin-bottom: 12px;"><b>Поздравляем! Настройка аккаунта завершена.</b></p><p style="margin-bottom: 12px;">Последний шаг: сгенерируйте API-ключ, чтобы проверить, что сервис работает корректно.</p><div style="padding: 8px 12px; background: #eff6ff; border-left: 3px solid #3b82f6; border-radius: 4px; font-size: 13px; margin-bottom: 12px;"><b>Зачем нужен API-ключ:</b><ul style="margin: 8px 0 0 16px;"><li>Это учётные данные для вызова AI-сервисов</li><li>Каждый ключ привязан к одной группе</li><li>Можно задавать квоту и срок действия</li><li>Поддерживается независимая статистика использования</li></ul></div><p style="margin-top: 16px; color: #10b981; font-weight: 600;">Нажмите "API Keys" в левом меню</p></div>'
      },
      createKey: {
        title: 'Создание ключа',
        description: '<div style="line-height: 1.7;"><p style="margin-bottom: 12px;">Нажмите кнопку, чтобы создать ваш первый API-ключ.</p><p style="padding: 8px 12px; background: #f0fdf4; border-left: 3px solid #10b981; border-radius: 4px; font-size: 13px;"><b>Совет:</b> сразу скопируйте и сохраните ключ после создания — он показывается только один раз</p><p style="margin-top: 12px; color: #10b981; font-weight: 600;">Нажмите кнопку "Создать ключ"</p></div>'
      },
      keyName: {
        title: '1. Имя ключа',
        description: '<div style="line-height: 1.7;"><p style="margin-bottom: 12px;">Задайте понятное имя для ключа.</p><p style="padding: 8px 12px; background: #f0fdf4; border-left: 3px solid #10b981; border-radius: 4px; font-size: 13px;"><b>Примеры:</b> "Test Key", "Production", "Mobile" и т.д.</p></div>',
        nextBtn: 'Далее'
      },
      keyGroup: {
        title: '2. Выбор группы',
        description: '<div style="line-height: 1.7;"><p style="margin-bottom: 12px;">Выберите группу, которую вы только что настроили.</p><div style="padding: 8px 12px; background: #eff6ff; border-left: 3px solid #3b82f6; border-radius: 4px; font-size: 13px; margin-bottom: 12px;"><b>Группа определяет:</b><ul style="margin: 8px 0 0 16px;"><li>Какие аккаунты может использовать этот ключ</li><li>Какой биллинговый коэффициент применяется</li><li>Будет ли ключ эксклюзивным</li></ul></div><p style="padding: 8px 12px; background: #f0fdf4; border-left: 3px solid #10b981; border-radius: 4px; font-size: 13px;"><b>Совет:</b> выберите тестовую группу, которую вы только что создали</p></div>',
        nextBtn: 'Далее'
      },
      keySubmit: {
        title: 'Генерация и копирование',
        description: '<div style="line-height: 1.7;"><p style="margin-bottom: 12px;">После нажатия "Создать" система сгенерирует полноценный API-ключ.</p><div style="padding: 8px 12px; background: #fee2e2; border-left: 3px solid #ef4444; border-radius: 4px; font-size: 13px; margin-bottom: 12px;"><b>Важно:</b><ul style="margin: 8px 0 0 16px;"><li>Ключ показывается только один раз — сразу скопируйте его</li><li>Если потеряете, придётся сгенерировать новый</li><li>Храните ключ безопасно и не передавайте другим</li></ul></div><div style="padding: 8px 12px; background: #f0fdf4; border-left: 3px solid #10b981; border-radius: 4px; font-size: 13px; margin-bottom: 12px;"><b>Дальше:</b><ul style="margin: 8px 0 0 16px;"><li>Скопируйте сгенерированный ключ sk-xxx</li><li>Используйте его в любом OpenAI-совместимом клиенте</li><li>Начинайте пользоваться AI-сервисами</li></ul></div><p style="margin-top: 12px; color: #10b981; font-weight: 600;">Нажмите кнопку "Создать"</p></div>'
      }
    },
    user: {
      welcome: {
        title: 'Добро пожаловать в Sub2API',
        description: '<div style="line-height: 1.8;"><p style="margin-bottom: 16px;">Здравствуйте! Добро пожаловать на платформу AI-сервисов Sub2API.</p><p style="margin-bottom: 12px;"><b>Быстрый старт:</b></p><ul style="margin-left: 20px; margin-bottom: 16px;"><li>Создайте API-ключ</li><li>Скопируйте ключ в своё приложение</li><li>Начните пользоваться AI-сервисами</li></ul><p style="color: #10b981; font-weight: 600;">Это займёт около одной минуты.</p></div>',
        nextBtn: 'Начать',
        prevBtn: 'Пропустить'
      },
      keyManage: {
        title: 'Управление API-ключами',
        description: '<div style="line-height: 1.7;"><p style="margin-bottom: 12px;">Здесь вы можете управлять всеми своими ключами доступа к API.</p><p style="padding: 8px 12px; background: #eff6ff; border-left: 3px solid #3b82f6; border-radius: 4px; font-size: 13px;"><b>Что такое API-ключ?</b><br/>API-ключ — это ваши учётные данные для доступа к AI-сервисам, своего рода ключ, который позволяет вашему приложению вызывать AI-возможности.</p><p style="margin-top: 12px; color: #10b981; font-weight: 600;">Нажмите, чтобы перейти на страницу ключей</p></div>'
      },
      createKey: {
        title: 'Создание нового ключа',
        description: '<div style="line-height: 1.7;"><p style="margin-bottom: 12px;">Нажмите кнопку, чтобы создать свой первый API-ключ.</p><p style="padding: 8px 12px; background: #f0fdf4; border-left: 3px solid #10b981; border-radius: 4px; font-size: 13px;"><b>Совет:</b> после создания ключ показывается только один раз, обязательно сразу скопируйте и сохраните его</p><p style="margin-top: 12px; color: #10b981; font-weight: 600;">Нажмите "Создать ключ"</p></div>'
      },
      keyName: {
        title: 'Имя ключа',
        description: '<div style="line-height: 1.7;"><p style="margin-bottom: 12px;">Задайте ключу понятное имя.</p><p style="padding: 8px 12px; background: #f0fdf4; border-left: 3px solid #10b981; border-radius: 4px; font-size: 13px;"><b>Примеры:</b> "My First Key", "For Testing" и т.д.</p></div>',
        nextBtn: 'Далее'
      },
      keyGroup: {
        title: 'Выбор группы',
        description: '<div style="line-height: 1.7;"><p style="margin-bottom: 12px;">Выберите сервисную группу, назначенную администратором.</p><p style="padding: 8px 12px; background: #eff6ff; border-left: 3px solid #3b82f6; border-radius: 4px; font-size: 13px;"><b>О группе:</b><br/>Разные группы могут иметь разное качество сервиса и разные тарифы, выбирайте по своим потребностям.</p></div>',
        nextBtn: 'Далее'
