package ai.greatfree.distributed.cnn.coordinator;

import java.io.IOException;

import org.greatfree.exceptions.NullClassConversionException;
import org.greatfree.exceptions.RemoteReadException;
import org.greatfree.util.Tools;

import ai.greatfree.distributed.rnn.RNNConfig;
import edu.greatfree.container.PeerProfile;
import edu.greatfree.exceptions.TaskAlreadyExistedException;

/**
 * Created: 09/01/2024, Bing Li
 */
final class StartCoordinator
{
	public static void main(String[] args) throws ClassNotFoundException, IOException, TaskAlreadyExistedException, RemoteReadException, InterruptedException
	{
		System.out.println("Starting the coordinator ...");
		Coordinator.TRAIN().start(PeerProfile.getAsyncPeerSpec(RNNConfig.TRAINING_COORDINATOR, RNNConfig.TRAINING_COORDINATOR_PORT, RNNConfig.REGISTRY_IP, RNNConfig.REGISTRY_PORT, new CoordinatorTask()), RNNConfig.TRAINING_ROOT);
		System.out.println("The coordinator started ...");
		
		int option = CoordMenuOptions.NO_OPTION;
		String optionStr;
		while (option != CoordMenuOptions.QUIT)
		{
			CoordinatorUI.printMenu();
			optionStr = Tools.INPUT.nextLine();
			try
			{
				option = Integer.parseInt(optionStr);
				System.out.println("Your choice: " + option);
				CoordinatorUI.execute(option);
			}
			catch (NumberFormatException | NullClassConversionException e)
			{
				option = CoordMenuOptions.NO_OPTION;
				System.out.println(CoordMenuItems.WRONG_OPTION);
			}
		}
		Coordinator.TRAIN().stop();
	}
}

