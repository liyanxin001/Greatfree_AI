package ai.greatfree.mlp;

import java.io.IOException;

import ai.djl.translate.TranslateException;

/**
 * Created: 08/18/2024, Bing Li
 */
final class ConciseDropoutMLP
{
	public static void main(String[] args) throws IOException, TranslateException
	{
		MLPUtils.conciseDropoutTrain(0.2f, 0.5f);
	}
}

