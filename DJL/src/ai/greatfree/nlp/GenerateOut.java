package ai.greatfree.nlp;

import java.util.Arrays;

import ai.greatfree.utils.RandomGenerator;

/**
 * Created: 08/24/2024, Bing Li
 */
final class GenerateOut
{
	public static void main(String[] args)
	{
		RandomGenerator generator = new RandomGenerator(Arrays.asList(new Double[] { 2.0, 3.0, 4.0 }));
		Integer[] generatorOutput = new Integer[10];
		for (int i = 0; i < 10; i++)
		{
			generatorOutput[i] = generator.draw();
		}
		System.out.println(Arrays.toString(generatorOutput));
	}
}

