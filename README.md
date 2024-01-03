<body>
  <h1>Web ML сервис</h1>
  <h3>Задача сервиса:</h3>
  <p>
    Необходимо спрогнозировать <ins>факт наличие у клиента просрочки в 90/ 90+ дней</ins>.
    Если отклик положительный, то банк отказывает данному человеку в получении кредита
  </p>
  <h3>Директория:</h3>
  <ul>
    <li> data - данные для обучения</li>
    <li> opt - дополнительные файлы</li>
    <li> weights - веса моделей</li>
    <li> EDA.ipynb - ноутбук с EDA</li>
    <li> features_importance.ipynb - ноутбук с Feature Importance модели</li>
    <li> train.ipynb - ноутбук с трейном и селектом модели</li>
    <li>cols_map.json - описание признаков</li>
    <li>requirements.txt - streamlit requirements</li>
    <li>service.py - web-сервис на базе streamlit</li>
  </ul>
  <h3>Model Selection:</h3>
  <p>
    Проведет тест следующих моделей:
  </p>
  <ul>
    <li>naive_bayes - GaussianNB, MultinomialNB, BernoulliNB, ComplementNB</li>
    <li>деревянные классификаторы - DecisionTreeClassifier, RandomForestClassifier, ExtraTreesClassifier</li>
    <li>линейные модели - LogisticRegression, LinearSVC, RidgeClassifier</li>
    <li>метрические модели - KNeighborsClassifier</li>
    <li>бустовые модели - GradientBoostingClassifier, CatBoostClassifier, LGBMClassifier, XGBClassifier, HistGradientBoostingClassifier</li>
    <li>дискриминантный анализ - LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis</li>
  </ul>
  <p>
    Функционал качества модели: <ins>f1</ins>
  </p>
</body>
