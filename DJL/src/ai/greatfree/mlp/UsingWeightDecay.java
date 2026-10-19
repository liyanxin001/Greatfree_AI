package ai.greatfree.mlp;

import java.io.IOException;

import ai.djl.translate.TranslateException;

/**
 * Created: 08/18/2024, Bing Li
 */
final class UsingWeightDecay
{
	public static void main(String[] args) throws IOException, TranslateException
	{
		MLPUtils.train(3f);
	}
}

