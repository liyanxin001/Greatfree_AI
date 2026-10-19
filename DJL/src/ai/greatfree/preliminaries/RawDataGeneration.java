package ai.greatfree.preliminaries;

import java.io.File;
import java.io.FileWriter;
import java.io.IOException;

/**
 * Created: 08/09/2024, Bing Li
 */
final class RawDataGeneration
{
	public static void main(String[] args) throws IOException
	{
		File file = new File("/Users/libing/Temp/house_tiny.csv");
		file.createNewFile();
		
		try (FileWriter writer = new FileWriter(file))
		{
			writer.write("NumRooms,Alley,Price\n");
			writer.write("NA,Pave,127500\n");
			writer.write("2,NA,106000\n");
			writer.write("4,NA,178100\n");
			writer.write("NA,NA,140000\n");
		}
	}
}
