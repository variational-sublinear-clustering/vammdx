/* Copyright (C) 2024 Machine Learning Lab of the University of Oldenburg. */
/* Licensed under the Academic Free License version 3.0                    */

#pragma once

#include <Eigen/Dense>
#include <stdexcept>

#include "Variational.h"
#include "Diagonal.h"

class DenoisingDiagonal
{
public:
    Diagonal &model;
    DenoisingDiagonal(Diagonal &);

    template <class Lmbd>
    void estimate_allocate(const Lmbd& lmbd) const;

    void estimate(cRef<Vector<>>, Ref<Vector<>>, const q_t &) const;

#ifdef CPPLIB_ENABLE_PYTHON_INTERFACE

    static void bind(pybind11::module_ &m);

#endif
};

//--------------------------------------------------------------------------------------------------------------------//

DenoisingDiagonal::DenoisingDiagonal(Diagonal &_model) : model(_model) {}

template <class Lmbd>
void DenoisingDiagonal::estimate_allocate(const Lmbd& lmbd) const
{    
    lmbd();
}

void DenoisingDiagonal::estimate(cRef<Vector<>>, Ref<Vector<>> x_reco, const q_t &q) const
{
    for (auto &[c, posterior] : q)
    {
        x_reco += posterior * model.M.row(c);
    }
}

#ifdef CPPLIB_ENABLE_PYTHON_INTERFACE
void DenoisingDiagonal::bind(pybind11::module_ &m)
{
    pybind11::class_<DenoisingDiagonal> DenoisingDiagonal_class_(m, "DenoisingDiagonal", pybind11::module_local());

    DenoisingDiagonal_class_.def(pybind11::init<Diagonal &>(), "model"_a);
}

#endif
