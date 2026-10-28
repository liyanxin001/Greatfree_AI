package ai.greatfree.distributed.cnn.child.registry;

/*    */ import java.util.HashSet;
/*    */ import java.util.Map;
/*    */ import java.util.Set;
/*    */ import java.util.concurrent.ConcurrentHashMap;
/*    */ import org.greatfree.util.UtilConfig;
/*    */ 
/*    */ 
/*    */ 
/*    */ 
/*    */ 
/*    */ 
/*    */ 
/*    */ class Clusters
/*    */ {
/* 17 */   private Map<String, Set<String>> clusterChildren = new ConcurrentHashMap<String, Set<String>>();
/*    */ 
/*    */ 
/*    */ 
/*    */ 
/*    */   
/* 23 */   private static Clusters instance = new Clusters();
/*    */ 
/*    */   
/*    */   public static Clusters SYSTEM() {
/* 27 */     if (instance == null) {
/*    */       
/* 29 */       instance = new Clusters();
/* 30 */       return instance;
/*    */     } 
/*    */ 
/*    */     
/* 34 */     return instance;
/*    */   }
/*    */ 
/*    */ 
/*    */   
/*    */   public void dispose() {
/* 40 */     this.clusterChildren.clear();
/* 41 */     this.clusterChildren = null;
/*    */   }
/*    */ 
/*    */ 
/*    */   
/*    */   public void joinCluster(String rootKey, String childKey) {
/* 47 */     if (!this.clusterChildren.containsKey(rootKey)) {
/*    */ 
/*    */       
/* 50 */       Set<String> children = new HashSet<String>();
/* 51 */       this.clusterChildren.put(rootKey, children);
/*    */     } 
/* 53 */     this.clusterChildren.get(rootKey).add(childKey);
/*    */   }
/*    */ 
/*    */   
/*    */   public void leaveCluster(String rootKey, String childKey) {
/* 58 */     if (this.clusterChildren.containsKey(rootKey))
/*    */     {
/* 60 */       this.clusterChildren.get(rootKey).remove(childKey);
/*    */     }
/*    */   }
/*    */ 
/*    */ 
/*    */   
/*    */   public Set<String> getChildKeys(String rootKey) {
/* 67 */     if (this.clusterChildren.containsKey(rootKey))
/*    */     {
/* 69 */       return this.clusterChildren.get(rootKey);
/*    */     }
/* 71 */     return UtilConfig.NO_KEYS;
/*    */   }
/*    */ }
