package ai.greatfree.linearregression;

import ai.djl.ndarray.NDList;

/**
 * Created: 08/15/2024, Bing Li
 */
class Updater
{
	public static void updater(NDList params, float lr, int batchSize)
	{
		LNTraining.sgd(params, lr, batchSize);
	}
}
