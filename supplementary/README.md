# Дополнительные материалы

Только агрегированные результаты актуальной статьи. Каждая строка описывает показатель, модель, фолд или группу, а не отдельного пациента. Идентификаторы, даты пациентов и индивидуальные прогнозы не включены. Таблицы с малыми группами содержат только согласованные сводные результаты, уже описанные в рукописи.

Основной расчет выполнен 27.09.2026, пороги повторно проверены 04.11.2026. Значения не пересчитывались при подготовке репозитория. Происхождение, преобразования и SHA-256 файлов записаны в [catalog.json](catalog.json). Исходные пути в каталоге относятся к закрытой рабочей папке автора и не доступны в публичном комплекте.

AP означает average precision. В исходных технических таблицах эта метрика называлась PR-AUC. Значение `<0.001` заменяет округленное `0.000`, а не означает новое вычисление p. В таблицах статистики `нет` и `да` обозначают единичное и рецидивирующее течение соответственно. Дробные значения TP/FP/FN/TN в OOF-таблицах получены усреднением по повторам, а не представляют количество пациентов в одном разбиении.

Пороги и ОШ разведочные: границы выбирались на той же когорте. Невложенные оценки финальных конфигураций не следует подменять ими основную вложенную оценку. Данные о синтетике, TabPFN и дополнительные анализы вне статьи не включены.

| Файл | Содержание | Строк |
|---|---|---:|
| [step1_recurrence_quantitative.csv](tables/step1_recurrence_quantitative.csv) | Сравнение количественных показателей по течению ДКА | 15 |
| [step1_recurrence_categorical.csv](tables/step1_recurrence_categorical.csv) | Сравнение категориальных показателей по течению ДКА | 31 |
| [step3_baseline.csv](tables/step3_baseline.csv) | Базовая логистическая регрессия | 1 |
| [step4_imputation.csv](tables/step4_imputation.csv) | Сравнение способов импутации | 3 |
| [step5_imbalance.csv](tables/step5_imbalance.csv) | Сравнение способов работы с дисбалансом | 3 |
| [step6_grid.csv](tables/step6_grid.csv) | Логистическая регрессия А, вложенная оценка | 2 |
| [step6_params_by_fold.csv](tables/step6_params_by_fold.csv) | Параметры ЛР А по внешним фолдам | 25 |
| [step7_methods.csv](tables/step7_methods.csv) | Пять процедур отбора и голосование | 7 |
| [step7_selection_frequency.csv](tables/step7_selection_frequency.csv) | Частоты отбора признаков по внешним фолдам | 24 |
| [step7_sets.csv](tables/step7_sets.csv) | Наборы признаков на всей обучающей части | 24 |
| [step8_models.csv](tables/step8_models.csv) | Вложенная оценка древесных моделей | 3 |
| [step8_boundaries.csv](tables/step8_boundaries.csv) | Выбранные параметры и края диапазонов поиска | 16 |
| [step8_choices.csv](tables/step8_choices.csv) | Частоты выбора категориальных гиперпараметров | 4 |
| [step8_convergence.csv](tables/step8_convergence.csv) | Сходимость поиска Optuna | 3 |
| [step9_models.csv](tables/step9_models.csv) | Основное сравнение моделей по вложенной CV | 4 |
| [step9_operating_points.csv](tables/step9_operating_points.csv) | Рабочие точки по OOF, пороги выбраны на тех же прогнозах | 12 |
| [step9_final_models.csv](tables/step9_final_models.csv) | Фиксированные конфигурации, оценка не вложена в выбор конфигурации | 4 |
| [step9_final_fixed_threshold.csv](tables/step9_final_fixed_threshold.csv) | OOF при зафиксированных порогах финальных конфигураций | 8 |
| [step10_ranks.csv](tables/step10_ranks.csv) | Ранги SHAP на всей обучающей части | 24 |
| [step10_top_frequency.csv](tables/step10_top_frequency.csv) | Бутстрэп-устойчивость попадания в топ-5 SHAP | 24 |
| [step10_fold_ranks.csv](tables/step10_fold_ranks.csv) | Ранги SHAP на валидационных частях внешних фолдов | 24 |
| [step10_fold_top_frequency.csv](tables/step10_fold_top_frequency.csv) | Частоты попадания в топ-5 SHAP по внешним фолдам | 24 |
| [step10_pairs.csv](tables/step10_pairs.csv) | Парные вклады SHAP случайного леса | 10 |
| [step10_rules.csv](tables/step10_rules.csv) | Агрегированные листья дерева правил | 3 |
| [step10_rules_params_by_fold.csv](tables/step10_rules_params_by_fold.csv) | Параметры дерева правил по внешним фолдам | 25 |
| [step10_rules_quality.csv](tables/step10_rules_quality.csv) | Вложенная оценка дерева правил | 2 |
| [step10_rules_stability.csv](tables/step10_rules_stability.csv) | Частоты использования признаков в дереве правил | 22 |
| [step10_summary.csv](tables/step10_summary.csv) | Сопоставление отбора и интерпретации | 24 |
| [step11_test.csv](tables/step11_test.csv) | Финальная оценка на тесте | 2 |
| [step11_thresholds.csv](tables/step11_thresholds.csv) | Сводные матрицы ошибок и рабочие точки финалистов | 8 |
| [thresholds.csv](tables/thresholds.csv) | Пороги признаков статьи | 4 |
| [threshold_pairs.csv](tables/threshold_pairs.csv) | Три сочетания дозы, включенные в статью | 3 |
| [final_parameters.csv](tables/final_parameters.csv) | Гиперпараметры двух финалистов без обученных коэффициентов и индивидуальных прогнозов | 27 |
