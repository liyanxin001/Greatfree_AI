package ai.greatfree.preliminaries;

import ai.djl.engine.Engine;
import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDManager;
import ai.djl.ndarray.types.Shape;
import ai.djl.training.GradientCollector;

/**
 * Created: 08/11/2024, Bing Li
 */
final class AutoDifferentiation
{
	public static void main(String[] args)
	{
		try (NDManager manager = NDManager.newBaseManager())
		{
			NDArray x = manager.arange(4f);
			x.setRequiresGradient(true);
			try (GradientCollector gc = Engine.getInstance().newGradientCollector())
			{
				NDArray y = x.dot(x).mul(2);
				System.out.println(y);
				gc.backward(y);
			}
			x.getGradient();
			System.out.println(x.getGradient().eq(x.mul(4)));

			System.out.println("---------------------------------------");

			try (GradientCollector gc = Engine.getInstance().newGradientCollector())
			{
				NDArray y = x.sum();
				System.out.println(y);
				gc.backward(y);
			}
			System.out.println(x.getGradient());

			System.out.println("---------------------------------------");

			try (GradientCollector gc = Engine.getInstance().newGradientCollector())
			{
				NDArray y = x.mul(x);
				System.out.println(y);
				gc.backward(y);
			}
			System.out.println(x.getGradient());

			System.out.println("---------------------------------------");

			try (GradientCollector gc = Engine.getInstance().newGradientCollector())
			{
				NDArray y = x.mul(x);
				System.out.println(y);
				NDArray u = y.stopGradient();
				System.out.println(u);
				NDArray z = u.mul(x);
				gc.backward(z);
				System.out.println(x.getGradient().eq(u));
			}

			System.out.println("---------------------------------------");

			try (GradientCollector gc = Engine.getInstance().newGradientCollector())
			{
				NDArray y = x.mul(x);
				System.out.println(y);
				y = x.mul(x);
				System.out.println(y);
				gc.backward(y);
				System.out.println(x.getGradient().eq(x.mul(2)));
			}

			System.out.println("---------------------------------------");

			NDArray a = manager.randomNormal(new Shape(1));
			System.out.println(a);
			a.setRequiresGradient(true);
			System.out.println(a);
			try (GradientCollector gc = Engine.getInstance().newGradientCollector())
			{
				NDArray d = f(a);
				System.out.println(d);
				gc.backward(d);
				System.out.println(a.getGradient().eq(d.div(a)));
			}
		}
	}

	@SuppressWarnings("resource")
	public static NDArray f(NDArray a)
	{
		NDArray b = a.mul(2);
		while (b.norm().getFloat() < 1000)
		{
			b = b.mul(2);
		}
		NDArray c;
		if (b.sum().getFloat() > 0)
		{
			c = b;
		}
		else
		{
			c = b.mul(100);
		}
		return c;
	}
}
