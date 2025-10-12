from model.predictors.BasePredictorBank import BasePredictorBank, BasePredictorBankConfig

class BasePredictorFauna4D(BasePredictorBank):
    def __init__(self, cfg: BasePredictorBankConfig):
        super().__init__(cfg)
        self.mean_feature = None

    def forward(self, total_iter=None, is_training=True, batch=None, bank_enc=None):
        """Use mean feature of sequence data instead of per batch mean"""
        assert self.mean_feature is not None
        images = batch[0]
        batch_size, num_frames, _, h0, w0 = images.shape
        images = images.reshape(batch_size * num_frames, *images.shape[2:])  # 0~1
        images_in = images * 2 - 1  # rescale to (-1, 1)
        batch_features = self.forward_frozen_ViT(images_in, bank_enc)
        _, embeddings, weights = self.retrieve_memory_bank(batch_features, batch)
        bank_embedding_model_input = [self.mean_feature, embeddings, weights]
        prior_shape = self.netShape.getMesh(total_iter=total_iter, jitter_grid=is_training, feats=self.mean_feature)
        return prior_shape, self.netDINO, bank_embedding_model_input