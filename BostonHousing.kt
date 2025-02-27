package smile

import smile.regression.GradientTreeBoost
import smile.data.formula.Formula
import smile.io.Read
import smile.validation.CrossValidation
import org.apache.commons.csv.CSVFormat

fun main() {
    val dsFileFormat = CSVFormat.DEFAULT.builder()
        .setHeader()
        .setSkipHeaderRecord(true)
        .setDelimiter(',')
        .build()

    val dataset = Read.csv("src/main/resources/BostonHousing.csv", dsFileFormat)

    println(dataset)

    val formula = Formula.lhs("medv")

    val res = CrossValidation.regression(
        10, formula, dataset,
        { formula, data -> GradientTreeBoost.fit(formula, data) }
    )

    println(res)
}
