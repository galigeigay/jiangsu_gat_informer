import xgboost as xgb
from xgboost.callback import LearningRateScheduler

from .baseclass import BaseModel
from api.io.logger import setup_logger

import numpy as np

logger = setup_logger(__name__)


class XGBModel(BaseModel):
    def __init__(self,
                 classification=False,
                 label_type='wind',
                 time_budget=120,
                 ):
        """XGBoost模型，基于FLAML自动调参

        Args:
            classification (bool, optional): 是否为分类模型
            label_type (str, optional): 标签类型，'wind' 或 'solar'，用于区分调参策略
            time_budget (int, optional): FLAML自动调参时间预算（秒）
        """
        super().__init__(class_name='XGBModel')
        self.label_type = label_type
        self.time_budget = time_budget
        self._model = None
        if classification:
            self._default_model = xgb.XGBClassifier(
                n_estimators=200,
                max_depth=4,
                learning_rate=0.3,
                tree_method='hist',
                gamma=0,
                min_child_weight=1,
                subsample=0.9,
                colsample_bytree=0.9,
                reg_alpha=0,
                reg_lambda=1,
                missing=-999,
                device="cuda"
            )
        else:
            self._default_model = xgb.XGBRegressor(
                n_estimators=500,
                max_depth=10,
                learning_rate=0.02,
                tree_method='hist',
                gamma=5,
                min_child_weight=10,
                subsample=0.6,
                colsample_bytree=0.6,
                reg_alpha=5,
                reg_lambda=0,
                missing=-999,
                device="cuda"
            )

    def train(self, **kwargs):
        feature = None
        label = None
        for k in kwargs:
            if 'data' in k or 'feat' in k or 'feature' in k:
                feature = kwargs[k]
            if 'label' in k or 'target' in k:
                label = kwargs[k]
            if 'weights' in k:
                weights = kwargs[k]

        # 尝试使用 FLAML 自动调参；若未安装或失败则回退到默认参数
        try:
            from flaml import AutoML

            automl = AutoML()

            budget = 300 if self.label_type == 'wind' else 180

            metric = 'mape' if self.label_type == 'wind' else 'rmse'
            settings = {
                "time_budget": budget,
                "metric": metric,
                "task": 'regression',
                "estimator_list": ['xgboost'],
                "log_file_name": f"flaml_{self.label_type}.log",
                "seed": 42,
                "early_stop": True,  # 启用早停，提高调参效率
            }
            if self.label_type == 'wind':
                settings["max_iter"] = 300
            else:
                settings["max_iter"] = 300

            automl.fit(X_train=feature, y_train=label.values.ravel(), **settings)
            flaml_model = automl.model
            if hasattr(flaml_model, 'model'):
                self._model = flaml_model.model
            elif hasattr(flaml_model, 'estimator'):
                self._model = flaml_model.estimator
            else:
                self._model = flaml_model
            logger.info(f"[FLAML] {self.label_type} 最优参数: {automl.best_estimator}")
            logger.info(f"[FLAML] {self.label_type} 最优{metric.upper()}: {automl.best_loss:.4f}")
            logger.info(f"[FLAML] {self.label_type} 实际迭代次数: {automl.best_iteration}")
        except Exception as e:
            logger.warning(f"[FLAML] 自动调参失败，回退到默认参数: {e}")
            self._model = self._default_model
            self._model.fit(feature, label)

    def predict(self, **kwargs):
        feature = None
        for k in kwargs:
            if 'data' in k or 'feat' in k or 'feature' in k:
                feature = kwargs[k]
        if self._model is None:
            raise RuntimeError("模型未训练或未加载")
        booster = self._model.get_booster()
        dmat = xgb.DMatrix(feature)
        rst = booster.predict(dmat)
        return rst

    def save(self, **kwargs):
        model_name = 'model'
        for k in kwargs:
            if 'model_name' in k:
                model_name = kwargs[k]
        if self._model is None:
            raise RuntimeError("模型未训练，无法保存")
        self._model.save_model(model_name)
        return

    def load(self, **kwargs):
        model_name = 'model'
        for k in kwargs:
            if 'model_name' in k:
                model_name = kwargs[k]
        if self._model is None:
            self._model = self._default_model
        self._model.load_model(model_name)
        return
