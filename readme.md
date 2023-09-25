#这个项目解决的问题是如何从一组查询的查询计划中还原整个数据库

我们利用多维直方图的思想。

每个查询计划可以分解为若干个空间中的轴对齐超立方体和轴对齐超立方体中点的数量的约束,可以表示为(Bucket,Cardinality)约束对.

我们根据约束对构造多维直方图,这显然是一个复杂度极高的问题.空间中超立方体相交可以产生指数级的不重叠子区域,再对这些区域施加线性规划,或者我们认为的最大熵分布条件,计算的代价非常高.

我们观察到,直方图中Bucket的数量太多是问题的根源,其实,我们未必需要那么多直方图.

空间中数据分布是稀疏的,很多区域是没有数据分布的,在这些区域构造许多细小的直方图没有意义.因此,我们构造直方图的原则是,每个Bucket至少包含K个数据,如果少于K个数据,那我们在不违反约束的条件下,将其划分到其他的,距离最近的Bucket中.

具体做法是,我们将空间中的约束按照密度进行排序,密度低的我们先进行检查.再依次添加高密度的约束.低密度约束中数据逐渐流向高密度相交区域,当低密度区没有数据时,我们删除低密度区.