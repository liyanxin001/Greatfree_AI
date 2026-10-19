package ai.greatfree.computation;

import java.io.FileInputStream;
import java.io.FileNotFoundException;
import java.io.FileOutputStream;
import java.io.IOException;

import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDList;
import ai.djl.ndarray.NDManager;
import ai.djl.util.Utils;

/**
 * Created: 08/19/2024, Bing Li
 */
final class LoadSaveTensors
{
	public static void main(String[] args) throws FileNotFoundException, IOException
	{
		String file = "/Users/libing/Temp/x-file";
		String file2 = "/Users/libing/Temp/x-list";
		try (NDManager manager = NDManager.newBaseManager())
		{
			NDArray x = manager.arange(4);
			System.out.println(x);
			try (FileOutputStream fos = new FileOutputStream(file))
			{
				fos.write(x.encode());
			}

			System.out.println("------------------------");
			NDArray x2;
			try (FileInputStream fis = new FileInputStream(file))
			{
				// We use the `Utils` method `toByteArray()` to read
				// from a `FileInputStream` and return it as a `byte[]`.
				x2 = NDArray.decode(manager, Utils.toByteArray(fis));
				System.out.println(x2);
			}

			System.out.println("------------------------");
			NDList list = new NDList(x, x2);
			System.out.println(list);
			try (FileOutputStream fos = new FileOutputStream(file2))
			{
				fos.write(list.encode());
			}

			System.out.println("------------------------");
			try (FileInputStream fis = new FileInputStream(file2))
			{
				list = NDList.decode(manager, Utils.toByteArray(fis));
				System.out.println(list);
			}
		}
	}
}
