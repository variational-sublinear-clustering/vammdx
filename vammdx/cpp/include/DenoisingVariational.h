/* Copyright (C) 2024 Machine Learning Lab of the University of Oldenburg. */
/* Licensed under the Academic Free License version 3.0                    */

#pragma once

#ifdef CPPLIB_ENABLE_PYTHON_INTERFACE
#include <pybind11/functional.h>
#endif

#include "Variational.h"

class DenoisingVariational
{
public:
    Variational &em;
    DenoisingVariational(Variational &);

    template <class Model>
    void estimate(cRef<Matrix<>>, Ref<Matrix<>>, Model &) const;

#ifdef CPPLIB_ENABLE_PYTHON_INTERFACE

    template <typename... Model>
    static void bind(py::module_ &m);
#endif
};

//--------------------------------------------------------------------------------------------------------------------//

DenoisingVariational::DenoisingVariational(Variational &_em) : em(_em) {}

template <class Model>
void DenoisingVariational::estimate(cRef<Matrix<>> X, Ref<Matrix<>> X_reco, Model &model) const
{
    size_t N = X.rows();
    X_reco.fill(0.0);
    model.model.auxiliary();
    #pragma omp parallel
    {
        model.estimate_allocate([&](auto &...args) -> auto {
            #pragma omp for
            for (size_t n = 0; n < N; n++)
            {
                model.estimate(X.row(n), X_reco.row(n), em.qs[n], args...);
            }
        });
    }
}

#ifdef CPPLIB_ENABLE_PYTHON_INTERFACE

template <typename... Model>
void DenoisingVariational::bind(py::module_ &m)
{
    py::class_<DenoisingVariational> DenoisingVariational_class_(m, "DenoisingVariational", py::module_local());

    /* Bindings specific to DenoisingVariational */

    DenoisingVariational_class_.def(py::init<Variational &>(), "em"_a);

    (DenoisingVariational_class_.def(
         "estimate",
         [](DenoisingVariational &self, cRef<Matrix<>> X, Ref<Matrix<>> X_reco, const Model &model) -> void
         {
             self.estimate(X, X_reco, model);
         },
         "X"_a.noconvert(), "X_reco"_a.noconvert(), "model"_a, R"(

        Parameters
        ----------

        Returns
        -------
        None
    )"),
     ...);
}
#endif
