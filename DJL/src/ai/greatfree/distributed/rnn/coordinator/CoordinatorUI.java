package ai.greatfree.distributed.rnn.coordinator;

import java.io.IOException;

import org.greatfree.exceptions.NullClassConversionException;
import org.greatfree.exceptions.RemoteReadException;

/**
 * Created: 09/01/2024, Bing Li
 */
final class CoordinatorUI
{
	public static void printMenu()
	{
		System.out.println(CoordMenuItems.MENU_HEAD);

		System.out.println(CoordMenuItems.DISTRIBUTE_TRAINING_DATA);
		System.out.println(CoordMenuItems.INITIALIZAE);
		System.out.println(CoordMenuItems.START_TRAINING);
		System.out.println(CoordMenuItems.PREDICT);
		System.out.println(CoordMenuItems.SHUTDOWN_CHILDREN);
		System.out.println(CoordMenuItems.SHUTDOWN_ROOT);
		System.out.println(CoordMenuItems.SHUTDOWN_REGISTRY_SERVER);
		
		System.out.println(CoordMenuItems.QUIT);
		System.out.println(CoordMenuItems.MENU_TAIL);
		System.out.println(CoordMenuItems.INPUT_PROMPT);
	}
	
	public static void execute(int option) throws ClassNotFoundException, IOException, RemoteReadException, NullClassConversionException, InterruptedException
	{
		switch (option)
		{
			case CoordMenuOptions.DISTRIBUTE_TRAINING_DATA:
				Coordinator.TRAIN().distributeTrainingData();
				break;
				
			case CoordMenuOptions.INITIALIZAE:
				Coordinator.TRAIN().initializeTraining();
				break;
				
			case CoordMenuOptions.START_TRAINING:
				Coordinator.TRAIN().startTraining();
				break;
				
			case CoordMenuOptions.PREDICT:
				Coordinator.TRAIN().predict();
				break;
				
			case CoordMenuOptions.SHUTDOWN_CHILDREN:
				Coordinator.TRAIN().shutdownChildren();
				break;
				
			case CoordMenuOptions.SHUTDOWN_ROOT:
				Coordinator.TRAIN().shutdownRoot();
				break;
				
			case CoordMenuOptions.SHUTDOWN_REGISTRY_SERVER:
				Coordinator.TRAIN().shutdownRegistryServer();
				break;
				
			case CoordMenuOptions.QUIT:
				break;
		}
	}
}

